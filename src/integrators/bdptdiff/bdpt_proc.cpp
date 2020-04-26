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

#include <mitsuba/core/statistics.h>
#include <mitsuba/core/sfcurve.h>
#include <mitsuba/bidir/util.h>
#include "bdpt_proc.h"
#include <cmath>

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                         Worker implementation                        */
/* ==================================================================== */

class BDPTDiffRenderer : public WorkProcessor {
	public:
		BDPTDiffRenderer(const BDPTConfiguration &config) : m_config(config) { 		
			Log(EInfo, "Creating BDPTDiffRenderer");
			m_reductor = NULL;
		}

		BDPTDiffRenderer(Stream *stream, InstanceManager *manager)
			: WorkProcessor(stream, manager), m_config(stream) {
			Log(EInfo, "Creating BDPTDiffRenderer");

			m_reductor = NULL;
		}

		virtual ~BDPTDiffRenderer() {
			// Release the reductor, if any.
			Log(EInfo, "Releasing BDPTDiffRenderer");
			if(m_reductor) {
				delete[] m_reductor;
			}
		}

		void serialize(Stream *stream, InstanceManager *manager) const {
			m_config.serialize(stream);
		}

		ref<WorkUnit> createWorkUnit() const {
			return new RectangularWorkUnit();
		}

		ref<WorkResult> createWorkResult() const {
			Log(EInfo, "createWorkResult DATA: %dx%d", m_reductorSize.x, m_reductorSize.y);
			if(!m_reductor){
				Log(EInfo, "m_reductor is NULL. Running in raw mode.");
			}
			
			return new BDPTDiffWorkResult(m_config, m_rfilter.get(),
					m_sampler.get(),
					Vector2i(m_config.blockSize), isFilmReductor, m_reductor, m_reductorSize);
		}

		void prepare() {
			Log(EInfo, "Preparing BDPTDiffRenderer");
			Scene *scene = static_cast<Scene *>(getResource("scene"));
			m_scene = new Scene(scene);
			m_sampler = static_cast<Sampler *>(getResource("sampler"));
			m_sensor = static_cast<Sensor *>(getResource("sensor"));
			m_rfilter = m_sensor->getFilm()->getReconstructionFilter();

			if(m_sensor->getFilm()->isReductor()){
				isFilmReductor = true;
				auto r = m_sensor->getFilm()->getReductor();
				m_reductorSize = m_sensor->getFilm()->getReductorSize();
				Log(EInfo, "DiffRenderer: %dx%d", m_reductorSize.x, m_reductorSize.y);
				if(r){
					m_reductor = new Float[m_reductorSize.x * m_reductorSize.y];
					memcpy(m_reductor, r, sizeof(Float) * m_reductorSize.x * m_reductorSize.y);
				} else {
					m_reductor = NULL;
					Log(EInfo, "DiffRenderer running in RAW mode. No reductor found");
				}
			} else {
				isFilmReductor = false;
				m_reductor = NULL;
				m_reductorSize = Vector2i(0,0);
			}

			m_scene->removeSensor(scene->getSensor());
			m_scene->addSensor(m_sensor);
			m_scene->setSensor(m_sensor);
			m_scene->setSampler(m_sampler);
			m_scene->wakeup(NULL, m_resources);
			m_scene->initializeBidirectional();
		}

		void process(const WorkUnit *workUnit, WorkResult *workResult, const bool &stop) {
			const RectangularWorkUnit *rect = static_cast<const RectangularWorkUnit *>(workUnit);
			BDPTDiffWorkResult *result = static_cast<BDPTDiffWorkResult *>(workResult);
			bool needsTimeSample = m_sensor->needsTimeSample();
			Float time = m_sensor->getShutterOpen();

			result->setOffset(rect->getOffset());
			result->setSize(rect->getSize());
			result->clear();
			m_hilbertCurve.initialize(TVector2<uint8_t>(rect->getSize()));

#if defined(MTS_DEBUG_FP)
			enableFPExceptions();
#endif

			Path emitterSubpath;
			Path sensorSubpath;

			/* Determine the necessary random walk depths based on properties of
			   the endpoints */
			int emitterDepth = m_config.maxDepth,
			    sensorDepth = m_config.maxDepth;

			/* Go one extra step if the sensor can be intersected */
			if (!m_scene->hasDegenerateSensor() && emitterDepth != -1)
				++emitterDepth;

			/* Go one extra step if there are emitters that can be intersected */
			if (!m_scene->hasDegenerateEmitters() && sensorDepth != -1)
				++sensorDepth;

			for (size_t i=0; i<m_hilbertCurve.getPointCount(); ++i) {
				Point2i offset = Point2i(m_hilbertCurve[i]) + Vector2i(rect->getOffset());
				m_sampler->generate(offset);

				for (size_t j = 0; j<m_sampler->getSampleCount(offset); j++) {
					if (stop)
						break;

					if (needsTimeSample)
						time = m_sensor->sampleTime(m_sampler->next1D());

					/* Start new emitter and sensor subpaths */
					emitterSubpath.initialize(m_scene, time, EImportance, m_pool);
					sensorSubpath.initialize(m_scene, time, ERadiance, m_pool);

					/* Perform a random walk using alternating steps on each path */
					Path::alternatingRandomWalkFromPixel(m_scene, m_sampler,
							emitterSubpath, emitterDepth, sensorSubpath,
							sensorDepth, offset, m_config.rrDepth, m_pool);

					evaluate(result, emitterSubpath, sensorSubpath);

					emitterSubpath.release(m_pool);
					sensorSubpath.release(m_pool);

					m_sampler->advance();
				}
			}

#if defined(MTS_DEBUG_FP)
			disableFPExceptions();
#endif

			/* Make sure that there were no memory leaks */
			//Assert(m_pool.unused());
		}

		/// Evaluate the contributions of the given eye and light paths
		void evaluate(BDPTDiffWorkResult *wr,
				Path &emitterSubpath, Path &sensorSubpath) {

			Point2 initialSamplePos = sensorSubpath.vertex(1)->getSamplePosition();
			Point2i initialDiscretePos = sensorSubpath.vertex(1)->getPositionSamplingRecord().uv_disc;

			const Scene *scene = m_scene;
			PathVertex tempEndpoint, tempSample;
			PathEdge tempEdge, connectionEdge;

			/* Compute the combined weights along the two subpaths */
			Spectrum *importanceWeights = (Spectrum *) alloca(emitterSubpath.vertexCount() * sizeof(Spectrum)),
				 *radianceWeights  = (Spectrum *) alloca(sensorSubpath.vertexCount()  * sizeof(Spectrum));

			importanceWeights[0] = radianceWeights[0] = Spectrum(1.0f);
			for (size_t i=1; i<emitterSubpath.vertexCount(); ++i)
				importanceWeights[i] = importanceWeights[i-1] *
					emitterSubpath.vertex(i-1)->weight[EImportance] *
					emitterSubpath.vertex(i-1)->rrWeight *
					emitterSubpath.edge(i-1)->weight[EImportance];

			for (size_t i=1; i<sensorSubpath.vertexCount(); ++i)
				radianceWeights[i] = radianceWeights[i-1] *
					sensorSubpath.vertex(i-1)->weight[ERadiance] *
					sensorSubpath.vertex(i-1)->rrWeight *
					sensorSubpath.edge(i-1)->weight[ERadiance];

			Spectrum sampleValue(0.0f);
			// X,Y,D -> I(X,Y) * S(X,Y,D)

			for (int s = (int) emitterSubpath.vertexCount()-1; s >= 0; --s) {
				/* Determine the range of sensor vertices to be traversed,
				   while respecting the specified maximum path length */
				int minT = std::max(2-s, m_config.lightImage ? 0 : 2),
				    maxT = (int) sensorSubpath.vertexCount() - 1;
				if (m_config.maxDepth != -1)
					maxT = std::min(maxT, m_config.maxDepth + 1 - s);

				for (int t = maxT; t >= minT; --t) {
					// -- Each of these inner loops represents a path.
					// We need to compute _ S(x) _ which is the sum of 
					// BSDF::eval_diff() / BSDF::eval() for every node along the path.

					PathVertex
						*vsPred = emitterSubpath.vertexOrNull(s - 1),
						*vtPred = sensorSubpath.vertexOrNull(t - 1),
						*vs = emitterSubpath.vertex(s),
						*vt = sensorSubpath.vertex(t);
					PathEdge
						*vsEdge = emitterSubpath.edgeOrNull(s - 1),
						*vtEdge = sensorSubpath.edgeOrNull(t - 1);

					RestoreMeasureHelper rmh0(vs), rmh1(vt);

					/* Will be set to true if direct sampling was used */
					bool sampleDirect = false;

					/* Stores the pixel position associated with this sample */
					Point2 samplePos = initialSamplePos;

					/* Allowed remaining number of ENull vertices that can
					   be bridged via pathConnect (negative=arbitrarily many) */
					int remaining = m_config.maxDepth - s - t + 1;

					/* Will receive the path weight of the (s, t)-connection */
					Spectrum value;

					/* Account for the terms of the measurement contribution
					   function that are coupled to the connection endpoints */
					if (vs->isEmitterSupernode()) {
						/* If possible, convert 'vt' into an emitter sample */
						if (!vt->cast(scene, PathVertex::EEmitterSample) || vt->isDegenerate())
							continue;

						value = radianceWeights[t] *
							vs->eval(scene, vsPred, vt, EImportance) *
							vt->eval(scene, vtPred, vs, ERadiance);
					}
					else if (vt->isSensorSupernode()) {
						/* If possible, convert 'vs' into an sensor sample */
						if (!vs->cast(scene, PathVertex::ESensorSample) || vs->isDegenerate())
							continue;

						/* Make note of the changed pixel sample position */
						if (!vs->getSamplePosition(vsPred, samplePos))
							continue;

						value = importanceWeights[s] *
							vs->eval(scene, vsPred, vt, EImportance) *
							vt->eval(scene, vtPred, vs, ERadiance);
					}
					else if (m_config.sampleDirect && ((t == 1 && s > 1) || (s == 1 && t > 1))) {
						/* s==1/t==1 path: use a direct sampling strategy if requested */
						if (s == 1) {
							if (vt->isDegenerate())
								continue;
							/* Generate a position on an emitter using direct sampling */
							value = radianceWeights[t] * vt->sampleDirect(scene, m_sampler,
									&tempEndpoint, &tempEdge, &tempSample, EImportance);
							if (value.isZero())
								continue;
							vs = &tempSample; vsPred = &tempEndpoint; vsEdge = &tempEdge;
							value *= vt->eval(scene, vtPred, vs, ERadiance);
							vt->measure = EArea;
						}
						else {
							if (vs->isDegenerate())
								continue;
							/* Generate a position on the sensor using direct sampling */
							value = importanceWeights[s] * vs->sampleDirect(scene, m_sampler,
									&tempEndpoint, &tempEdge, &tempSample, ERadiance);
							if (value.isZero())
								continue;
							vt = &tempSample; vtPred = &tempEndpoint; vtEdge = &tempEdge;
							value *= vs->eval(scene, vsPred, vt, EImportance);
							vs->measure = EArea;
						}

						sampleDirect = true;
					}
					else {
						/* Can't connect degenerate endpoints */
						if (vs->isDegenerate() || vt->isDegenerate())
							continue;

						value = importanceWeights[s] * radianceWeights[t] *
							vs->eval(scene, vsPred, vt, EImportance) *
							vt->eval(scene, vtPred, vs, ERadiance);

						/* Temporarily force vertex measure to EArea. Needed to
						   handle BSDFs with diffuse + specular components */
						vs->measure = vt->measure = EArea;
					}

					/* Attempt to connect the two endpoints, which could result in
					   the creation of additional vertices (index-matched boundaries etc.) */
					int interactions = remaining; // backup
					if (value.isZero() || !connectionEdge.pathConnectAndCollapse(
								scene, vsEdge, vs, vt, vtEdge, interactions))
						continue;

					/* Account for the terms of the measurement contribution
					   function that are coupled to the connection edge */
					if (!sampleDirect)
						value *= connectionEdge.evalCached(vs, vt, PathEdge::EGeneralizedGeometricTerm);
					else
						value *= connectionEdge.evalCached(vs, vt, PathEdge::ETransmittance |
								(s == 1 ? PathEdge::ECosineRad : PathEdge::ECosineImp));

					if (sampleDirect) {
						/* A direct sampling strategy was used, which generated
						   two new vertices at one of the path ends. Temporarily
						   modify the path to reflect this change */
						if (t == 1)
							sensorSubpath.swapEndpoints(vtPred, vtEdge, vt);
						else
							emitterSubpath.swapEndpoints(vsPred, vsEdge, vs);
					}

					/* Compute the multiple importance sampling weight */
					Float miWeight = Path::miWeight(scene, emitterSubpath, &connectionEdge,
							sensorSubpath, s, t, m_config.sampleDirect, m_config.lightImage);

					// TODO: Check integer cast if it's correct.
					// int posX = static_cast<int>(initialSamplePos.x);
					// int posY = static_cast<int>(initialSamplePos.y);

					std::map<int, Spectrum> partialDiffs;
					std::map<std::tuple<int, int, int>, Spectrum> stSplitDiffs;

					// S_t(x)
					for (int _t = 2; _t <= t; _t++) {
						if (_t == t && s == 0) continue; // Degenerate connection.
						const BSDF* bsdf = sensorSubpath.vertex(_t)->getIntersection().getBSDF();
						if (!bsdf->isDifferentiable()) continue;

						Intersection its = sensorSubpath.vertex(_t)->getIntersection();

						Point predP = sensorSubpath.vertex(_t-1)->getPosition(),
						      succP = (_t != t ) ? sensorSubpath.vertex(_t+1)->getPosition() : emitterSubpath.vertex(s)->getPosition();

						Vector wi = normalize(predP - its.p),
						       wo = normalize(succP - its.p);

						BSDFSamplingRecord bRec(its, its.toLocal(wi),
								its.toLocal(wo), ERadiance);
						//sensorSubpath.vertex(_t)->get
						Spectrum f = bsdf->eval(bRec, ESolidAngle);
						DifferentialList fprime = bsdf->eval_diff(bRec, ESolidAngle);//.at(m_config.diffParameter);

						//if(fprime.size() != 3) 
						//	Log(EError, "The sparse differential module currently does not support differntials");

						if(fprime.size() == 0)
							continue;

						for(std::map<int, Spectrum>::iterator it = fprime.begin(); it != fprime.end(); ++it) {
							//v.push_back(it->first);

							int posD = it->first;
							// Accumulate the value.
							//sx += ( f != Spectrum(0.0) ) ? fprime / f : Spectrum(0.0);
							//if(it->second.abs().average() > 3.0) {
							//	it->second = it->second / (it->second.abs().average() / 3.0);
							//}

							//std::tuple<int, int, int> t = std::make_tuple(posX, posY, posD);
							std::tuple<int, int, int> k = std::make_tuple(s, t, posD);
							if(partialDiffs.count(posD) == 0) partialDiffs[posD] = Spectrum(0.0);
							if(stSplitDiffs.count(k) == 0) stSplitDiffs[k] = Spectrum(0.0);

							Spectrum temp = ((f.average() > 1e-10) ? (it->second / f.average()) : Spectrum(0.0)) * value.average() * miWeight;
							partialDiffs[posD] += temp;
							//stSplitDiffs[k] += ((f != Spectrum(0.0)) ? (it->second / f.average()) : Spectrum(0.0)) * value.average() * miWeight;

							//if(miWeight * value.average() * (it->second.average() / f.average()) > 300 || !std::isfinite(temp.average())) {
							if(
								!std::isfinite(temp.average())
							){
								
								std::cout << "Pos: " << initialDiscretePos.x << "," << initialDiscretePos.y << std::endl;
								std::cout << "final-grad: " << temp.toString() << std::endl;
								std::cout << "Value avg: " << value.toString() << " MiWeight: " << miWeight << " at t=" << t << ", s=" << s << ", _t=" << _t << std::endl;
								std::cout << "F: " << f.toString() << " value/F: " << (value / f).toString() << std::endl;
								std::cout << "dF/F: " << (f != Spectrum(0.0) ? (it->second / f.average()) : Spectrum(0.0)).toString() << " at t=" << t << ", s=" << s << ", _t=" << _t << std::endl;
								std::cout << "it->first:= "<< it->first << " it->second:= " << it->second.toString() << " at t=" << t << ", s=" << s << ", _t=" << _t << std::endl;
								std::cout << "wi: " << wi.toString() << " wo: " << wo.toString() << std::endl;
								std::cout << "Lwi: " << its.toLocal(wi).toString() << " Lwo: " << its.toLocal(wo).toString() << std::endl;
								std::cout << "LH: " << its.toLocal((wi + wo)/2).toString() << std::endl;
								std::cout << "Lgradient: " << its.toLocal(Vector(it->second[0], it->second[1], it->second[2])).toString() << std::endl;
								std::cout << "pred: " << predP.toString() << std::endl;
								std::cout << "this: " << its.p.toString() << std::endl;
								std::cout << "succ: " << succP.toString() << std::endl << std::endl;
							}

							auto hnvec = its.toLocal((wi + wo)/2);
							hnvec /= hnvec.length();
							if (m_config.angularHistogram){
								wr->putHistogramSample(s, t, hnvec.z, dot(hnvec, wi), value);
							}
							//cout << it->first << "\n";
						}

					}

					// S_s(x)
					for (int _s = 2; _s <= s; _s++) {
						if (_s == s && t == 0) continue; // Degenerate connection.
						const BSDF* bsdf = emitterSubpath.vertex(_s)->getIntersection().getBSDF();
						if (!bsdf->isDifferentiable()) continue;

						Intersection its = emitterSubpath.vertex(_s)->getIntersection();

						Point predP = emitterSubpath.vertex(_s - 1)->getPosition(),
						      succP = (_s != s) ? emitterSubpath.vertex(_s + 1)->getPosition() : sensorSubpath.vertex(t)->getPosition();


						Vector wi = normalize(predP - its.p),
						       wo = normalize(succP - its.p);

						BSDFSamplingRecord bRec(its, its.toLocal(wi),
								its.toLocal(wo), EImportance);

						//sensorSubpath.vertex(_t)->get
						Spectrum f = bsdf->eval(bRec, ESolidAngle);
						DifferentialList fprime = bsdf->eval_diff(bRec, ESolidAngle);//.at(m_config.diffParameter);

						if(fprime.size() == 0) 
							continue;

						for(std::map<int, Spectrum>::iterator it = fprime.begin(); it != fprime.end(); ++it) {
							//v.push_back(it->first);

							int posD = it->first;
							//if(it->second.abs().average() > 3.0) {
							//	it->second = it->second / (it->second.abs().average() / 3.0);
							//}
							// Accumulate the value.
							//sx += ( f != Spectrum(0.0) ) ? fprime / f : Spectrum(0.0);

							//std::tuple<int, int, int> t = std::make_tuple(posX, posY, posD);

							std::tuple<int, int, int> k = std::make_tuple(s, t, posD);
							if(partialDiffs.count(posD) == 0) partialDiffs[posD] = Spectrum(0.0);
							if(stSplitDiffs.count(k) == 0) stSplitDiffs[k] = Spectrum(0.0);

							Spectrum temp = ((f.average() > 1e-10) ? (it->second / f.average()) : Spectrum(0.0)) * value.average() * miWeight;
							partialDiffs[posD] += temp;

							//stSplitDiffs[k] += ((f.average() > 1e-5) ? (it->second / f.average()) : Spectrum(0.0)) * value.average() * miWeight;

							//std::cout << "Value avg: " << value.average() << " MiWeight: " << miWeight << " at t=" << t << ", s=" << s << ", _s=" << _s << std::endl;
							//std::cout << "dF/F: " << (f != Spectrum(0.0) ? (it->second / f.average()) : Spectrum(0.0)).toString() << " at t=" << t << ", s=" << s << ", _s=" << _s << std::endl;
							//std::cout << "it->first:= "<< it->first << " it->second:= " << it->second.toString() << " at t=" << t << ", s=" << s << ", _s=" << _s << std::endl;

							//if(miWeight * value.average() * (it->second.average() / f.average()) > 300 || !std::isfinite(temp.average())) {
							if(
								!std::isfinite(temp.average())
							) {
								std::cout << "Pos: " << initialDiscretePos.x << "," << initialDiscretePos.y << std::endl;
								std::cout << "final-grad: " << temp.toString() << std::endl;
								std::cout << "Value avg: " << value.toString() << " MiWeight: " << miWeight << " at t=" << t << ", s=" << s << ", _s=" << _s << std::endl;
								std::cout << "F: " << f.toString() << " value/F: " << (value / f).toString() << std::endl;
								std::cout << "dF/F: " << (f != Spectrum(0.0) ? (it->second / f.average()) : Spectrum(0.0)).toString() << " at t=" << t << ", s=" << s << ", _s=" << _s << std::endl;
								std::cout << "it->first:= "<< it->first << " it->second:= " << it->second.toString() << " at t=" << t << ", s=" << s << ", _s=" << _s << std::endl;
								std::cout << "wi: " << wi.toString() << " wo: " << wo.toString() << std::endl;
								std::cout << "Lwi: " << its.toLocal(wi).toString() << " Lwo: " << its.toLocal(wo).toString() << std::endl;
								std::cout << "LH: " << its.toLocal((wi + wo)/2).toString() << std::endl;
								std::cout << "Lgradient: " << its.toLocal(Vector(it->second[0], it->second[1], it->second[2])).toString() << std::endl;
								std::cout << "pred: " << predP.toString() << std::endl;
								std::cout << "this: " << its.p.toString() << std::endl;
								std::cout << "succ: " << succP.toString() << std::endl << std::endl;
							}
							auto hnvec = its.toLocal((wi + wo)/2);
							hnvec /= hnvec.length();

							if (m_config.angularHistogram){
								wr->putHistogramSample(s, t, hnvec.z, dot(hnvec, wi), value);
							}
						}
					}

					if (sampleDirect) {
						/* Now undo the previous change */
						if (t == 1)
							sensorSubpath.swapEndpoints(vtPred, vtEdge, vt);
						else
							emitterSubpath.swapEndpoints(vsPred, vsEdge, vs);
					}

					/* Determine the pixel sample position when necessary */
					if (vt->isSensorSample() && !vt->getSamplePosition(vs, samplePos))
						continue;

					// Compute the S(x) here.

					//Spectrum sx(0.0);
					for(auto p : partialDiffs) {
						if(std::isinf(p.second[0]) || std::isinf(p.second[1]) || std::isinf(p.second[2])) {
							Log(EError, "Detected illegal value at %f, %f at %d", initialSamplePos.x, initialSamplePos.y, p.first);
						}
						if(t >= 2 && !(m_config.pathFiltering && (m_config.tFilter != t || m_config.sFilter != s))) {
							Float pixelPdf = static_cast<Float>(m_sampler->getSampleCount(initialDiscretePos));
							//if(p.first > 81)
								//Log(EInfo, "Putting sample %d (t=%d,s=%d): %f,%f,%f", p.first, t, s, p.second[0], p.second[1], p.second[2]);
							wr->putSample(
								std::make_tuple(initialSamplePos.x, initialSamplePos.y, p.first),
								p.second,
								pixelPdf);
							//if(p.first != 0) {
							//	Log(EInfo, "non-0 p value: %d", p.first);
							//}
						}
						//else
						//    wr->putSample(std::make_tuple(samplePos.x, samplePos.y, p.first), p.second);
					}

#if BDPT_DEBUG == 1
					/* When the debug mode is on, collect samples
					   separately for each sampling strategy. Note: the
					   following piece of code artificially increases the
					   exposure of longer paths */
					Spectrum splatValue = value * (m_config.showWeighted
							? miWeight : 1.0f);// * std::pow(2.0f, s+t-3.0f));
					wr->putDebugSample(s, t, samplePos, splatValue);
#endif

					//sx = sx.abs();
					//sx = -sx;
					//if(sx > 0.0f) {
					//    Log(
					//}
					//sx = Spectrum(1.0f);
					//if (t >= 2)
					//    sampleValue += value * miWeight * sx;

					//else
					//wr->putLightSample(samplePos, value * miWeight * sx);
				}
			}

		}

		ref<WorkProcessor> clone() const {
			return new BDPTDiffRenderer(m_config);
		}

		MTS_DECLARE_CLASS()
	private:
			ref<Scene> m_scene;
			ref<Sensor> m_sensor;
			ref<Sampler> m_sampler;
			ref<ReconstructionFilter> m_rfilter;
			Float* m_reductor;
			Vector2i m_reductorSize;
			MemoryPool m_pool;
			BDPTConfiguration m_config;
			HilbertCurve2D<uint8_t> m_hilbertCurve;
			bool isFilmReductor;
};


/* ==================================================================== */
/*                           Parallel process                           */
/* ==================================================================== */

BDPTDiffProcess::BDPTDiffProcess(const RenderJob *parent, RenderQueue *queue,
		const BDPTConfiguration &config) :
	BlockedRenderProcess(parent, queue, config.blockSize), m_config(config) {
		m_refreshTimer = new Timer();
	}

ref<WorkProcessor> BDPTDiffProcess::createWorkProcessor() const {
	return new BDPTDiffRenderer(m_config);
}

void BDPTDiffProcess::develop() {
	if (!m_config.lightImage)
		return;
	//LockGuard lock(m_resultMutex);

	/*const ImageBlock *lightImage = m_result->getLightImage();
	  m_film->setBitmap(m_result->getImageBlock()->getBitmap());
	  m_film->addBitmap(lightImage->getBitmap(), 1.0f / m_config.sampleCount);
	  m_refreshTimer->reset();
	  m_queue->signalRefresh(m_parent);*/
}

void BDPTDiffProcess::processResult(const WorkResult *wr, bool cancelled) {
	if (cancelled)
		return;
	const BDPTDiffWorkResult *result = static_cast<const BDPTDiffWorkResult *>(wr);
	ReductorImageBlock *block = const_cast<ReductorImageBlock*>(result->getImageBlock());
	LockGuard lock(m_resultMutex);
	m_progress->update(++m_resultCount);
	/*if (m_config.lightImage) {
	  const ImageBlock *lightImage = m_result->getLightImage();
	  m_result->put(result);
	  if (m_parent->isInteractive()) {
	// Modify the finished image block so that it includes the light image contributions,
	//   which creates a more intuitive preview of the rendering process. This is
	//   not 100% correct but doesn't matter, as the shown image will be properly re-developed
	//   every 2 seconds and once more when the rendering process finishes

	Float invSampleCount = 1.0f / m_config.sampleCount;
	const Bitmap *sourceBitmap = lightImage->getBitmap();
	Bitmap *destBitmap = block->getBitmap();
	int borderSize = block->getBorderSize();
	Point2i offset = block->getOffset();
	Vector2i size = block->getSize();

	for (int y=0; y<size.y; ++y) {
	const Float *source = sourceBitmap->getFloatData()
	+ (offset.x + (y+offset.y) * sourceBitmap->getWidth()) * SPECTRUM_SAMPLES;
	Float *dest = destBitmap->getFloatData()
	+ (borderSize + (y + borderSize) * destBitmap->getWidth()) * (SPECTRUM_SAMPLES + 2);

	for (int x=0; x<size.x; ++x) {
	Float weight = dest[SPECTRUM_SAMPLES + 1] * invSampleCount;
	for (int k=0; k<SPECTRUM_SAMPLES; ++k)
	 *dest++ += *source++ * weight;
	 dest += 2;
	 }
	 }
	 }
	 }*/

	// Reweigh the image intensities.
	// TODO: Incorrect weighing. need to use a weight map.
	std::map<int, Spectrum> gradients = block->getData();
	std::map<int, Spectrum> newgradients = block->getData();
	Float invSampleCount = 1.0f / m_config.sampleCount;
	for (auto gitem : gradients)
		newgradients[gitem.first] = gitem.second * invSampleCount;

	block->setData(newgradients);
	m_film->put(reinterpret_cast<const ImageBlock*>(block));

	/* Re-develop the entire image every two seconds if partial results are
	   visible (e.g. in a graphical user interface). This only applies when
	   there is a light image. */

	// TODO: There is no light image.

	//bool developFilm = m_config.lightImage &&
	//    (m_parent->isInteractive() && m_refreshTimer->getMilliseconds() > 2000);

	// TODO: Potential problems because of the incorrect recast.
	m_queue->signalWorkEnd(m_parent, reinterpret_cast<const ImageBlock*>(result->getImageBlock()), false);

	//if (developFilm)
	//    develop();
}

void BDPTDiffProcess::bindResource(const std::string &name, int id) {
	BlockedRenderProcess::bindResource(name, id);
	if (name == "sensor" && m_config.lightImage) {
		/* If needed, allocate memory for the light image */
		m_result = new BDPTDiffWorkResult(m_config, NULL, NULL, m_film->getCropSize());
		m_result->clear();
	}
}

	MTS_IMPLEMENT_CLASS_S(BDPTDiffRenderer, false, WorkProcessor)
	MTS_IMPLEMENT_CLASS(BDPTDiffProcess, false, BlockedRenderProcess)
	MTS_NAMESPACE_END
