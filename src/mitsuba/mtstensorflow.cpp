/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/core/platform.h>

// Mitsuba's "Assert" macro conflicts with Xerces' XSerializeEngine::Assert(...).
// This becomes a problem when using a PCH which contains mitsuba/core/logger.h
#if defined(Assert)
# undef Assert
#endif
#include <xercesc/parsers/SAXParser.hpp>
#include <mitsuba/core/sched_remote.h>
#include <mitsuba/core/sstream.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/appender.h>
#include <mitsuba/core/sshstream.h>
#include <mitsuba/core/shvector.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/render/renderjob.h>
#include <mitsuba/render/scenehandler.h>
#include <fstream>
#include <stdexcept>
#include <boost/algorithm/string.hpp>
#include <sstream>

#if defined(__WINDOWS__)
#include <mitsuba/core/getopt.h>
#include <winsock2.h>
#else
#include <signal.h>
#endif

#ifdef __WINDOWS__
#include <io.h>
#include <ws2tcpip.h>
#else
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#define INVALID_SOCKET -1
#define SOCKET int
#endif

#define CONN_BACKLOG 5
using XERCES_CPP_NAMESPACE::SAXParser;

using namespace mitsuba;
static SOCKET sock = INVALID_SOCKET;
static bool running = true;

#if defined(__WINDOWS__)
BOOL CtrlHandler(DWORD type) {
    switch (type) {
        case CTRL_C_EVENT:
            running = false;
            if (sock) {
                closesocket(sock);
                sock = INVALID_SOCKET;
            }
            return TRUE;
        default:
            return FALSE;
    }
}

#else
/* Catch Ctrl+C, SIGKILL */
void sigterm_handler(int) {
    SLog(EInfo, "Caught signal - shutting down..");
    running = false;

    /* The next signal will immediately terminate the
       program. (a precaution for hung processes) */
    signal(SIGTERM, SIG_DFL);
    signal(SIGINT, SIG_DFL);
}

/* Collect zombie processes */
void collect_zombies(int s) {
    while (waitpid(-1, NULL, WNOHANG) > 0);
}
#endif
void help() {
    cout <<  "Mitsuba version " << Version(MTS_VERSION).toStringComplete()
         << ", Copyright (c) " MTS_YEAR " Wenzel Jakob" << endl;
    cout <<  "Usage: mtstensorflow [options] <One or more scene XML files>" << endl;
    cout <<  "Options/Arguments:" << endl;
    cout <<  "   -h          Display this help text" << endl << endl;
    cout <<  "   -D key=val  Define a constant, which can referenced as \"$key\" in the scene" << endl << endl;
    cout <<  "   -o fname    Write the output image to the file denoted by \"fname\"" << endl << endl;
    cout <<  "   -a p1;p2;.. Add one or more entries to the resource search path" << endl << endl;
    cout <<  "   -p count    Override the detected number of processors. Useful for reducing" << endl;
    cout <<  "               the load or creating scheduling-only nodes in conjunction with"  << endl;
    cout <<  "               the -c and -s parameters, e.g. -p 0 -c host1;host2;host3,..." << endl << endl;
    cout <<  "   -q          Quiet mode - do not print any log messages to stdout" << endl << endl;
    cout <<  "   -c hosts    Network rendering: connect to mtssrv instances over a network." << endl;
    cout <<  "               Requires a semicolon-separated list of host names of the form" << endl;
    cout <<  "                       host.domain[:port] for a direct connection" << endl;
    cout <<  "                 or" << endl;
    cout <<  "                       user@host.domain[:path] for a SSH connection (where" << endl;
    cout <<  "                       \"path\" denotes the place where Mitsuba is checked" << endl;
    cout <<  "                       out -- by default, \"~/mitsuba\" is used)" << endl << endl;
    cout <<  "   -s file     Connect to additional Mitsuba servers specified in a file" << endl;
    cout <<  "               with one name per line (same format as in -c)" << endl<< endl;
    cout <<  "   -j count    Simultaneously schedule several scenes. Can sometimes accelerate" << endl;
    cout <<  "               rendering when large amounts of processing power are available" << endl;
    cout <<  "               (e.g. when running Mitsuba on a cluster. Default: 1)" << endl << endl;
    cout <<  "   -n name     Assign a node name to this instance (Default: host name)" << endl << endl;
    cout <<  "   -x          Skip rendering of files where output already exists" << endl << endl;
    cout <<  "   -r sec      Write (partial) output images every 'sec' seconds" << endl << endl;
    cout <<  "   -b res      Specify the block resolution used to split images into parallel" << endl;
    cout <<  "               workloads (default: 32). Only applies to some integrators." << endl << endl;
    cout <<  "   -v          Be more verbose (can be specified twice)" << endl << endl;
    cout <<  "   -L level    Explicitly specify the log level (trace/debug/info/warn/error)" << endl << endl;
    cout <<  "   -w          Treat warnings as errors" << endl << endl;
    cout <<  "   -z          Disable progress bars" << endl << endl;
    cout <<  " For documentation, please refer to http://www.mitsuba-renderer.org/docs.html" << endl;
}

ref<RenderQueue> renderQueue = NULL;

#if !defined(__WINDOWS__)
/* Handle the hang-up signal and write a partially rendered image to disk */
void signalHandler(int signal) {
    if (signal == SIGHUP && renderQueue.get()) {
        renderQueue->flush();
    } else if (signal == SIGFPE) {
        SLog(EWarn, "Caught a floating-point exception!");

        #if defined(MTS_DEBUG_FP)
        /* Generate a core dump! */
        abort();
        #endif
    }
}
#endif

class FlushThread : public Thread {
public:
    FlushThread(int timeout) : Thread("flush"),
        m_flag(new WaitFlag()),
        m_timeout(timeout) { }

    void run() {
        while (!m_flag->get()) {
            m_flag->wait(m_timeout * 1000);
            renderQueue->flush();
        }
    }

    void quit() {
        m_flag->set(true);
        join();
    }
private:
    ref<WaitFlag> m_flag;
    int m_timeout;
};

int mitsuba_app(int argc, char **argv) {
    int optchar;
    char *end_ptr = NULL;
//    short listenPort;
    try {
        /* Default settings */
        int nprocs_avail = getCoreCount(), nprocs = nprocs_avail, 
                listenPort = MTS_DEFAULT_PORT;
        int numParallelScenes = 1;
        std::string nodeName = getHostName(),
                    networkHosts = "", destFile="";
        bool quietMode = false, progressBars = true, skipExisting = false;
        ELogLevel logLevel = EInfo;
        ref<FileResolver> fileResolver = Thread::getThread()->getFileResolver();
        bool treatWarningsAsErrors = false;
        std::map<std::string, std::string, SimpleStringOrdering> parameters;
        std::vector<std::string> paramNames;
        std::string hostName = getFQDN();
        bool hostNameSet = false;
        int blockSize = 32;
        int flushTimer = -1;
        

        if (argc < 2) {
            help();
            return 0;
        }

        optind = 1;
        /* Parse command-line arguments */
        while ((optchar = getopt(argc, argv, "a:c:D:s:j:n:o:r:b:p:L:l:i:qhzvtwx")) != -1) {
            switch (optchar) {
                case 'a': {
                        std::vector<std::string> paths = tokenize(optarg, ";");
                        for (int i=(int) paths.size()-1; i>=0; --i)
                            fileResolver->prependPath(paths[i]);
                    }
                    break;
                case 'i':
                    hostName = optarg;
                    hostNameSet = true;
                    break;
                case 'c':
                    networkHosts = networkHosts + std::string(";") + std::string(optarg);
                    break;
                case 'w':
                    treatWarningsAsErrors = true;
                    break;
                case 'D': {
                        std::vector<std::string> param = tokenize(optarg, "=");
                        if (param.size() != 2)
                            SLog(EError, "Invalid parameter specification \"%s\"", optarg);
                        
                        parameters[param[0]] = param[1];
                        SLog(EInfo, "%s\n", optarg);
                        SLog(EInfo, "Got a param %s", param.at(0).c_str());
                        paramNames.push_back(param[0]);
                    }
                    break;
                case 's': {
                        std::ifstream is(optarg);
                        if (is.fail())
                            SLog(EError, "Could not open host file!");
                        std::string host;
                        while (is >> host) {
                            if (host.length() < 1 || host.c_str()[0] == '#')
                                continue;
                            networkHosts = networkHosts + std::string(";") + host;
                        }
                    }
                    break;
                case 'n':
                    nodeName = optarg;
                    break;
                case 'o':
                    destFile = optarg;
                    break;
                case 'v':
                    if (logLevel != EDebug)
                        logLevel = EDebug;
                    else
                        logLevel = ETrace;
                    break;
                case 'l':
                    if (!strcmp("s", optarg)) {
                        listenPort = -1;
                        quietMode = true;
                    } else {
                        listenPort = strtol(optarg, &end_ptr, 10);
                        if (*end_ptr != '\0')
                            SLog(EError, "Could not parse the port number");
                    }
                    break;
                case 'L': {
                        std::string arg = boost::to_lower_copy(std::string(optarg));
                        if (arg == "trace")
                            logLevel = ETrace;
                        else if (arg == "debug")
                            logLevel = EDebug;
                        else if (arg == "info")
                            logLevel = EInfo;
                        else if (arg == "warn")
                            logLevel = EWarn;
                        else if (arg == "error")
                            logLevel = EError;
                        else
                            SLog(EError, "Invalid log level!");
                    }
                    break;
                case 'x':
                    skipExisting = true;
                    break;
                case 'p':
                    nprocs = strtol(optarg, &end_ptr, 10);
                    if (*end_ptr != '\0')
                        SLog(EError, "Could not parse the processor count!");
                    break;
                case 'j':
                    numParallelScenes = strtol(optarg, &end_ptr, 10);
                    if (*end_ptr != '\0')
                        SLog(EError, "Could not parse the parallel scene count!");
                    break;
                case 'r':
                    flushTimer = strtol(optarg, &end_ptr, 10);
                    if (*end_ptr != '\0')
                        SLog(EError, "Could not parse the '-r' parameter argument!");
                    break;
                case 'b':
                    blockSize = strtol(optarg, &end_ptr, 10);
                    if (*end_ptr != '\0')
                        SLog(EError, "Could not parse the block size!");
                    if (blockSize < 2 || blockSize > 128)
                        SLog(EError, "Invalid block size (should be in the range 2-128)");
                    break;
                case 'z':
                    progressBars = false;
                    break;
                case 'q':
                    quietMode = true;
                    break;
                case 'h':
                default:
                    help();
                    return 0;
            }
        }

        ProgressReporter::setEnabled(progressBars);

        /* Initialize OpenMP */
        Thread::initializeOpenMP(nprocs);

        /* Configure the logging subsystem */
        ref<Logger> log = Thread::getThread()->getLogger();
        log->setLogLevel(logLevel);
        log->setErrorLevel(treatWarningsAsErrors ? EWarn : EError);

        /* Disable the default appenders */
        for (size_t i=0; i<log->getAppenderCount(); ++i) {
            Appender *appender = log->getAppender(i);
            if (appender->getClass()->derivesFrom(MTS_CLASS(StreamAppender)))
                log->removeAppender(appender);
        }

        log->addAppender(new StreamAppender(formatString("mitsuba.%s.log", nodeName.c_str())));
        if (!quietMode)
            log->addAppender(new StreamAppender(&std::cout));

        SLog(EInfo, "Mitsuba version %s, Copyright (c) " MTS_YEAR " Wenzel Jakob",
                Version(MTS_VERSION).toStringComplete().c_str());

        /* Configure the scheduling subsystem */
        Scheduler *scheduler = Scheduler::getInstance();
        bool useCoreAffinity = nprocs == nprocs_avail;
        for (int i=0; i<nprocs; ++i)
            scheduler->registerWorker(new LocalWorker(useCoreAffinity ? i : -1,
                formatString("wrk%i", i)));
        std::vector<std::string> hosts = tokenize(networkHosts, ";");

        /* Establish network connections to nested servers */
        for (size_t i=0; i<hosts.size(); ++i) {
            const std::string &hostName = hosts[i];
            ref<Stream> stream;

            if (hostName.find("@") == std::string::npos) {
                int port = MTS_DEFAULT_PORT;
                std::vector<std::string> tokens = tokenize(hostName, ":");
                if (tokens.size() == 0 || tokens.size() > 2) {
                    SLog(EError, "Invalid host specification '%s'!", hostName.c_str());
                } else if (tokens.size() == 2) {
                    port = strtol(tokens[1].c_str(), &end_ptr, 10);
                    if (*end_ptr != '\0')
                        SLog(EError, "Invalid host specification '%s'!", hostName.c_str());
                }
                stream = new SocketStream(tokens[0], port);
            } else {
                std::string path = "~/mitsuba"; // default path if not specified
                std::vector<std::string> tokens = tokenize(hostName, "@:");
                if (tokens.size() < 2 || tokens.size() > 3) {
                    SLog(EError, "Invalid host specification '%s'!", hostName.c_str());
                } else if (tokens.size() == 3) {
                    path = tokens[2];
                }
                std::vector<std::string> cmdLine;
                cmdLine.push_back(formatString("bash -c 'cd %s; . setpath.sh; mtssrv -ls'", path.c_str()));
                stream = new SSHStream(tokens[0], tokens[1], cmdLine);
            }
            try {
                scheduler->registerWorker(new RemoteWorker(formatString("net%i", i), stream));
            } catch (std::runtime_error &e) {
                if (hostName.find("@") != std::string::npos) {
#if defined(__WINDOWS__)
                    SLog(EWarn, "Please ensure that passwordless authentication "
                        "using plink.exe and pageant.exe is enabled (see the documentation for more information)");
#else
                    SLog(EWarn, "Please ensure that passwordless authentication "
                        "is enabled (e.g. using ssh-agent - see the documentation for more information)");
#endif
                }
                throw e;
            }
        }

        scheduler->start();

#if !defined(__WINDOWS__)
            /* Initialize signal handlers */
            /*struct sigaction sa;
            sa.sa_handler = signalHandler;
            sigemptyset(&sa.sa_mask);
            sa.sa_flags = 0;
            if (sigaction(SIGHUP, &sa, NULL))
                SLog(EError, "Could not install a custom signal handler!");
            if (sigaction(SIGFPE, &sa, NULL))
                SLog(EError, "Could not install a custom signal handler!");*/
#endif

        /* Prepare for parsing scene descriptions */
        SAXParser* parser = new SAXParser();
        fs::path schemaPath = fileResolver->resolveAbsolute("data/schema/scene.xsd");

        /* Check against the 'scene.xsd' XML Schema */
        parser->setDoSchema(true);
        parser->setValidationSchemaFullChecking(true);
        parser->setValidationScheme(SAXParser::Val_Always);
        parser->setExternalNoNamespaceSchemaLocation(schemaPath.c_str());

        /* Set the handler */
        SceneHandler *handler = new SceneHandler(parameters);
        parser->setDoNamespaces(true);
        parser->setDocumentHandler(handler);
        parser->setErrorHandler(handler);

        renderQueue = new RenderQueue();

        ref<FlushThread> flushThread;
        if (flushTimer > 0) {
            flushThread = new FlushThread(flushTimer);
            flushThread->start();
        }

// SERVER PART.

        /* Allocate a socket of the proper type (IPv4/IPv6) */
        struct addrinfo hints, *servinfo, *p = NULL;
        memset(&hints, 0, sizeof(struct addrinfo));
        hints.ai_family = AF_UNSPEC;
        hints.ai_flags = AI_PASSIVE;
        hints.ai_socktype = SOCK_STREAM;
        char portName[8];
        int rv, one = 1;
        sock = INVALID_SOCKET;

        snprintf(portName, sizeof(portName), "%i", listenPort);
        if ((rv = getaddrinfo(hostNameSet ? hostName.c_str() : NULL, portName, &hints, &servinfo)) != 0)
            SLog(EError, "Error in getaddrinfo(%s:%i): %s", hostName.c_str(), listenPort, gai_strerror(rv));

        for (p = servinfo; p != NULL; p = p->ai_next) {
            /* Allocate a socket */
            sock = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
            if (sock == -1)
                SocketStream::handleError("none", "socket");

            /* Avoid "bind: socket already in use" */
            if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char *) &one, sizeof(int)) < 0)
                SocketStream::handleError("none", "setsockopt");

            /* Bind the socket to the port number */
            if (bind(sock, p->ai_addr, (socklen_t) p->ai_addrlen) == -1) {
                SocketStream::handleError("none", formatString("bind(%s:%i)", hostName.c_str(), listenPort), EError);
#if defined(__WINDOWS__)
                closesocket(sock);
#else
                close(sock);
#endif
                continue;
            }
            break;
        }

        if (p == NULL)
            SLog(EError, "Failed to bind to port %i!", listenPort);
        freeaddrinfo(servinfo);

        if (listen(sock, CONN_BACKLOG) == -1)
            SocketStream::handleError("none", "bind");
        SLog(EInfo, "Enter mtssrv -h for more options");

#if defined(__WINDOWS__)
        SLog(EInfo, "%s: Listening on port %i.. Send Ctrl-C to stop.", hostName.c_str(), listenPort);
#else
        /* Avoid zombies processes */
        struct sigaction sa;
        sa.sa_handler = collect_zombies;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;

        if (sigaction(SIGCHLD, &sa, NULL) == -1)
            SLog(EError, "Error in sigaction(): %s!", strerror(errno));

        sa.sa_handler = sigterm_handler;
        sa.sa_flags = 0; // we want SIGINT/SIGTERM to interrupt accept()

        if (sigaction(SIGTERM, &sa, NULL) == -1)
            SLog(EError, "Error in sigaction(): %s!", strerror(errno));
        if (sigaction(SIGINT, &sa, NULL) == -1)
            SLog(EError, "Error in sigaction(): %s!", strerror(errno));

        /* Ignore SIGPIPE */
        signal(SIGPIPE, SIG_IGN);

        SLog(EInfo, "%s: Listening on port %i.. Send Ctrl-C or SIGTERM to stop.", hostName.c_str(), listenPort);
#endif
        
        int connectionIndex = 0;
        int jobIdx = 0;

        /* Wait for connections */
        while (running) {
            SLog(EInfo, "Waiting for connections.");
            socklen_t addrlen = sizeof(sockaddr_storage);
            struct sockaddr_storage sockaddr;
            memset(&sockaddr, 0, addrlen);

            SOCKET newSocket = accept(sock, (struct sockaddr *) &sockaddr, &addrlen);
            if (newSocket == INVALID_SOCKET) {
                SLog(EInfo, "Received invalid connection.");
#if defined(__WINDOWS__)
                if (!running)
                    break;
#else
                if (errno == EINTR)
                    continue;
#endif
                SocketStream::handleError("none", "accept", EWarn);
                continue;
            }

            // Wait for a signal from tensorflow op.
            ref<SocketStream> ss = new SocketStream(newSocket);
            SLog(EInfo, "Got a connection.");  

            // Find number of parameters.
            short numvals;
            read(newSocket, &numvals, sizeof(short));
            SLog(EInfo, "Num parameters: %d", numvals);

            // Read and set all the parameters.
            // The order is important.
            for(int i = 0; i < numvals; i++ ){
                //float val = ss->readSingle();
                float val;
                int bytes = read(newSocket, &val, sizeof(val));

                std::stringstream sst;
                if(floor(val) == val) {
                    int ival = static_cast<int>(floor(val));
                    SLog(EInfo, "%i/%i %s:%i (int cast)", i, numvals, paramNames.at(i).c_str(), ival);
                    sst << ival;
                } else {
                    SLog(EInfo, "%i/%i %s:%f", i, numvals, paramNames.at(i).c_str(), val);
                    sst << val;
                }

                parameters[paramNames.at(i)] = sst.str();
            }
            
            SLog(EInfo, "Rendering.. ");
            SceneHandler* handler = new SceneHandler(parameters);
            parser->setDoNamespaces(true);
            parser->setDocumentHandler(handler);
            parser->setErrorHandler(handler);

            fs::path
                filename = fileResolver->resolve(argv[optind]),
                filePath = fs::absolute(filename).parent_path(),
                baseName = filename.stem();

            ref<FileResolver> frClone = fileResolver->clone();
            frClone->prependPath(filePath);
            Thread::getThread()->setFileResolver(frClone);

            SLog(EInfo, "Parsing scene description from \"%s\" ..", argv[optind]);

            parser->parse(filename.c_str());
            ref<Scene> scene = handler->getScene();

            SLog(EInfo, "Done parsing \"%s\" ..", argv[optind]);

            scene->setSourceFile(filename);
            scene->setDestinationFile(destFile.length() > 0 ?
                fs::path(destFile) : (filePath / baseName));
            scene->setBlockSize(blockSize);

            SLog(EInfo, "Dispatching render job \"%s\" ..", argv[optind]);

            if (scene->destinationExists() && skipExisting)
                continue;

            SLog(EInfo, "Dispatching render job \"%s\" ..", argv[optind]);

            // Dispatch render job.
            ref<RenderJob> thr = new RenderJob(formatString("ren%i", jobIdx++),
                scene, renderQueue, -1, -1, -1, true, flushTimer > 0);
            thr->start();

            SLog(EInfo, "Dispatched render job \"%s\" ..", argv[optind]);


            renderQueue->waitLeft(numParallelScenes-1);
            delete handler;
        }
// SERVER PART END.

        /* Wait for all render processes to finish */
        renderQueue->waitLeft(0);
        if (flushThread)
            flushThread->quit();
        renderQueue = NULL;

        delete handler;
        delete parser;

        Statistics::getInstance()->printStats();
    } catch (const std::exception &e) {
        std::cerr << "Caught a critical exception: " << e.what() << endl;
        return -1;
    } catch (...) {
        std::cerr << "Caught a critical exception of unknown type!" << endl;
        return -1;
    }

    return 0;
}

int mts_main(int argc, char **argv) {
    /* Initialize the core framework */
    Class::staticInitialization();
    Object::staticInitialization();
    PluginManager::staticInitialization();
    Statistics::staticInitialization();
    Thread::staticInitialization();
    Logger::staticInitialization();
    FileStream::staticInitialization();
    Spectrum::staticInitialization();
    Bitmap::staticInitialization();
    Scheduler::staticInitialization();
    SHVector::staticInitialization();
    SceneHandler::staticInitialization();

#if defined(__WINDOWS__)
    /* Initialize WINSOCK2 */
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2,2), &wsaData))
        SLog(EError, "Could not initialize WinSock2!");
    if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2)
        SLog(EError, "Could not find the required version of winsock.dll!");
#endif

#if defined(__LINUX__) || defined(__OSX__)
    /* Correct number parsing on some locales (e.g. ru_RU) */
    setlocale(LC_NUMERIC, "C");
#endif

    int retval = mitsuba_app(argc, argv);

    /* Shutdown the core framework */
    SceneHandler::staticShutdown();
    SHVector::staticShutdown();
    Scheduler::staticShutdown();
    Bitmap::staticShutdown();
    Spectrum::staticShutdown();
    FileStream::staticShutdown();
    Logger::staticShutdown();
    Thread::staticShutdown();
    Statistics::staticShutdown();
    PluginManager::staticShutdown();
    Object::staticShutdown();
    Class::staticShutdown();

#if defined(__WINDOWS__)
    /* Shut down WINSOCK2 */
    WSACleanup();
#endif

    return retval;
}

#if !defined(__WINDOWS__)
int main(int argc, char **argv) {
    return mts_main(argc, argv);
}
#endif
