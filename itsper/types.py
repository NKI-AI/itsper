from enum import Enum


class ItsperAnnotationTypes(str, Enum):
    """
    Enum class for the different types of annotations that the ITSP AI can generate.
    """

    MI_REGION = "Most invasive region"
    TUMORBED = "Tumorbed"


class ItsperWsiExtensions(Enum):
    TIFF = ".tiff"
    SVS = ".svs"
    MRXS = ".mrxs"


class ItsperAnnotationExtensions(str, Enum):
    """
    Enum class for the different types of annotations that the ITSP AI can process.
    """

    JSON = "json"


class ItsperInferenceExtensions(str, Enum):
    """
    Enum class for the different types of images that the ITSP AI can process.
    """

    TIFF = "tiff"


class ITSPScoringSheetHeaders(int, Enum):
    """
    Enum class for the different headers in the scoring sheet.
    """

    SLIDESCORE_ID = 0
    SLIDE_ID = 1
    USER = 2
    ITSP_SCORE = 4
