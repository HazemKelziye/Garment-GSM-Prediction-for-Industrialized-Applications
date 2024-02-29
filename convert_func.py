import pandas as pd


def convert_fraction_to_float(value):
    """
    This function is designed for converting the Yarn Count feature
    :param value: Can be a string or numeric type; expected to be a fraction in the form of a string (e.g., '1/2') or
                  other numeric representations for conversion.
    :return: A numeric type (float or other types depending on the input). Returns pd.NA if the input cannot be
             converted to a numeric type due to format or data errors.
    """
    try:
        if '/' in str(value):
            numerator, denominator = value.split('/')
            return float(numerator) / float(denominator)
        else:
            return pd.to_numeric(value, errors='coerce')
    except ValueError:
        return pd.NA
