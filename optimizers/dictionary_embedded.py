import json
import os
import sys


def embedMERLDictionary(
        dictionaryFile,
        sceneFile,
        outSceneFile,
        outParameterFile,
        outPaddingFile,
        adaptiveSampling=False,
        outAdaptiveParametersFile=None,
        ignoreBSDFMode=False,
        placeholder="@@@BSDF-PLACEHOLDER@@@"):

    # Load XML as a generic file.
    xmldata = open(sceneFile, "r").read()

    dictionary = json.load(open(dictionaryFile, "r"))

    bsdfString = "<bsdf type=\"diffwrapper\" id=\"" + dictionary["id"] + "\">\n"

    bsdfString += "<bsdf type=\"dj_merl\">\n"

    parameterList = []
    sampleParameterList = []

    bsdfString += "<string name=\"filename\" value=\"/tmp/tabular-bsdf-$meshSlot.binary\"/>\n"
    bsdfString += "</bsdf>\n"

    bsdfString += "<boolean name=\"differentiable\" value=\"true\"/>\n"
    bsdfString += "</bsdf>\n"

    # Replace the placeholder
    xmldata = xmldata.replace(placeholder, bsdfString)

    open(outSceneFile, "w").write(xmldata)

    # Write the parameter list as well.
    json.dump(parameterList, open(outParameterFile, "w"))
    json.dump([dictionary["undifferentiables"]], open(outPaddingFile,"w"))

    if adaptiveSampling:
        assert(outAdaptiveParametersFile is not None)
        json.dump(sampleParameterList, open(outAdaptiveParametersFile, "w"))

# Embeds a BSDF dictionary into a mitsuba XML scene file.
def embedDictionary(
        dictionaryFile,
        sceneFile,
        outSceneFile,
        outParameterFile,
        outPaddingFile,
        adaptiveSampling=False,
        outAdaptiveParametersFile=None,
        ignoreBSDFMode=False,
        placeholder="@@@BSDF-PLACEHOLDER@@@"):

    # Load XML as a generic file.
    xmldata = open(sceneFile, "r").read()

    dictionary = json.load(open(dictionaryFile, "r"))

    if "type" in dictionary and dictionary["type"] == "tabular":
        return embedMERLDictionary(
                    dictionaryFile,
                    sceneFile,
                    outSceneFile,
                    outParameterFile,
                    outPaddingFile,
                    adaptiveSampling,
                    outAdaptiveParametersFile,
                    ignoreBSDFMode, placeholder)

    bsdfString = "<bsdf type=\"diffwrapper\" id=\""+dictionary["id"]+"\">\n"
    if not adaptiveSampling or (ignoreBSDFMode):
        bsdfString += "<bsdf type=\"mixturebsdf\">\n"
    else:
        bsdfString += "<bsdf type=\"mixturesampledbsdf\">\n"

    parameterList = []
    sampleParameterList = []

    bsdfString += "<string name=\"weights\" value=\""
    for i in range(len(dictionary["elements"])):
        parameterList.append("weight" + format(i).zfill(4))
        bsdfString += "$weight" + format(i).zfill(4) + ","

    # Handle old version of the dictionary.
    if "undifferentiables" not in dictionary:
        dictionary["undifferentiables"] = len(dictionary["subdifferentiables"])

    for i in range(len(dictionary["subdifferentiables"]) - dictionary["undifferentiables"]):
        parameterList.append(dictionary["subdifferentiables"][i])

    bsdfString = bsdfString[:-1] # Get rid of the last comma.
    bsdfString += "\"/>\n"

    if adaptiveSampling and (not ignoreBSDFMode):
        # Add a 'samples' parameter to the BSDF.
        bsdfString += "<string name=\"samples\" value=\""
        for i in range(len(dictionary["elements"])):
            sampleParameterList.append("sampleWeight" + format(i).zfill(4))
            bsdfString += "$sampleWeight" + format(i).zfill(4) + ","
        bsdfString = bsdfString[:-1] # Get rid of the last comma.
        bsdfString += "\"/>\n"

    for bsdf in dictionary["elements"]:
        bsdfString += "\n<bsdf type=\"" + bsdf["type"] + "\">\n"
        if "alpha" in bsdf:
            bsdfString += "\t<float name=\"alpha\" value=\"" + format(bsdf["alpha"]) + "\"/>\n"
        if "distribution" in bsdf:
            bsdfString += "\t<string name=\"distribution\" value=\"" + bsdf["distribution"] + "\"/>\n"

        bsdfString += "\t<boolean name=\"differentiable\" value=\"true\"/>\n"

        if "eta" in bsdf:
            bsdfString += "\t<spectrum name=\"eta\" value=\"" + format(bsdf["eta"]) + "\"/>\n"
        if "k" in bsdf:
            bsdfString += "\t<spectrum name=\"k\" value=\"" + format(bsdf["k"]) + "\"/>\n"
        if "reflectance" in bsdf:
            bsdfString += "\t<spectrum name=\"reflectance\" value=\"" + format(bsdf["reflectance"]) + "\"/>\n"

        bsdfString += "</bsdf>\n"
    bsdfString += "</bsdf>\n"
    bsdfString += "<boolean name=\"differentiable\" value=\"true\"/>\n"
    bsdfString += "</bsdf>\n"

    # Replace the placeholder
    xmldata = xmldata.replace(placeholder, bsdfString)

    open(outSceneFile, "w").write(xmldata)

    # Write the parameter list as well.
    json.dump(parameterList, open(outParameterFile, "w"))
    json.dump([dictionary["undifferentiables"]], open(outPaddingFile,"w"))

    if adaptiveSampling:
        assert(outAdaptiveParametersFile is not None)
        json.dump(sampleParameterList, open(outAdaptiveParametersFile, "w"))