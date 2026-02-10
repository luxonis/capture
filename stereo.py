import depthai as dai


def _apply_common(stereo, initial_config, settings):
    stereo.setRectification(True)
    initial_config.algorithmControl.enableExtended = settings["extendedDisparity"]
    initial_config.algorithmControl.leftRightCheckThreshold = 10
    
    initial_config.algorithmControl.disparityShift = 0
    initial_config.algorithmControl.numInvalidateEdgePixels = 0

    initial_config.censusTransform.noiseThresholdOffset = 1
    initial_config.censusTransform.noiseThresholdScale = 1

    initial_config.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)

    initial_config.postProcessing.spatialFilter.enable = False
    initial_config.postProcessing.spatialFilter.alpha = 0.5
    initial_config.postProcessing.spatialFilter.delta = 3
    initial_config.postProcessing.spatialFilter.holeFillingRadius = 2
    initial_config.postProcessing.spatialFilter.numIterations = 1

    initial_config.postProcessing.temporalFilter.enable = False
    initial_config.postProcessing.temporalFilter.alpha = 0.4
    initial_config.postProcessing.temporalFilter.delta = 3
    initial_config.postProcessing.temporalFilter.persistencyMode = dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_4

    initial_config.postProcessing.speckleFilter.enable = False
    initial_config.postProcessing.speckleFilter.speckleRange = 50

    initial_config.postProcessing.thresholdFilter.minRange = 0
    initial_config.postProcessing.thresholdFilter.maxRange = 65000

    initial_config.postProcessing.decimationFilter.decimationFactor = 1
    initial_config.postProcessing.decimationFilter.decimationMode = dai.StereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.PIXEL_SKIPPING

    initial_config.postProcessing.brightnessFilter.minBrightness = 0
    initial_config.postProcessing.brightnessFilter.maxBrightness = 255


def _apply_rvc2(stereo, initial_config):
    stereo.setLeftRightCheck(True)
    initial_config.algorithmControl.enableSubpixel = True
    initial_config.algorithmControl.subpixelFractionalBits = 5
    initial_config.costMatching.confidenceThreshold = 55
    initial_config.costMatching.disparityWidth = dai.StereoDepthConfig.CostMatching.DisparityWidth.DISPARITY_96
    initial_config.costMatching.enableCompanding = False
    initial_config.costMatching.linearEquationParameters.alpha = 0
    initial_config.costMatching.linearEquationParameters.beta = 2
    initial_config.costMatching.linearEquationParameters.threshold = 127

    initial_config.costAggregation.divisionFactor = 1
    initial_config.costAggregation.horizontalPenaltyCostP1 = 250
    initial_config.costAggregation.horizontalPenaltyCostP2 = 500
    initial_config.costAggregation.verticalPenaltyCostP1 = 250
    initial_config.costAggregation.verticalPenaltyCostP2 = 500

    initial_config.censusTransform.enableMeanMode = True
    initial_config.censusTransform.kernelMask = 0
    initial_config.censusTransform.kernelSize = dai.StereoDepthConfig.CensusTransform.KernelSize.AUTO
    initial_config.censusTransform.threshold = 0


def _apply_rvc4(initial_config):
    initial_config.algorithmControl.enableSwLeftRightCheck = True
    initial_config.costMatching.enableSwConfidenceThresholding = False
    
    initial_config.postProcessing.adaptiveMedianFilter.enable = True
    initial_config.postProcessing.adaptiveMedianFilter.confidenceThreshold = 200

    initial_config.postProcessing.holeFilling.enable = True
    initial_config.postProcessing.holeFilling.highConfidenceThreshold = 210
    initial_config.postProcessing.holeFilling.fillConfidenceThreshold = 200
    initial_config.postProcessing.holeFilling.minValidDisparity = 1
    initial_config.postProcessing.holeFilling.invalidateDisparities = True

    initial_config.costAggregation.p1Config.enableAdaptive = True
    initial_config.costAggregation.p1Config.defaultValue = 11
    initial_config.costAggregation.p1Config.edgeValue = 10
    initial_config.costAggregation.p1Config.smoothValue = 22
    initial_config.costAggregation.p1Config.edgeThreshold = 15
    initial_config.costAggregation.p1Config.smoothThreshold = 5

    initial_config.costAggregation.p2Config.enableAdaptive = True
    initial_config.costAggregation.p2Config.defaultValue = 33
    initial_config.costAggregation.p2Config.edgeValue = 22
    initial_config.costAggregation.p2Config.smoothValue = 63

    initial_config.confidenceMetrics.occlusionConfidenceWeight = 20
    initial_config.confidenceMetrics.motionVectorConfidenceWeight = 4
    initial_config.confidenceMetrics.motionVectorConfidenceThreshold = 1
    initial_config.confidenceMetrics.flatnessConfidenceWeight = 8
    initial_config.confidenceMetrics.flatnessConfidenceThreshold = 2
    initial_config.confidenceMetrics.flatnessOverride = False


def setup_stereo(pipeline, settings, platform):
    """
    Set up stereo depth node with parameters for the given platform.

    :param pipeline: DepthAI pipeline object
    :param settings: Dictionary containing stereo settings (e.g. extendedDisparity)
    :param platform: dai.Platform.RVC2 or dai.Platform.RVC4
    :return: Configured StereoDepth node
    """
    stereo = pipeline.create(dai.node.StereoDepth)
    initial_config = stereo.initialConfig

    # Option 1: use a profile preset (e.g. DEFAULT, ROBOTICS, FAST_ACCURACY, FAST_DENSITY)
    # stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
    # return stereo

    # Option 2: configure parameters yourself (common + platform-specific below)
    _apply_common(stereo, initial_config, settings)

    if platform == dai.Platform.RVC2:
        _apply_rvc2(stereo, initial_config)
    elif platform == dai.Platform.RVC4:
        _apply_rvc4(initial_config)

    return stereo
