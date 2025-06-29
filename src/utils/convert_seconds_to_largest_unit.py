def convert_seconds_to_largest_unit(seconds, scaled_data=None):
    """
    Convert a given time in seconds to the largest possible unit (minutes, hours, days, or years).
    Optionally, scale a provided data array based on the conversion.

    Parameters:
        seconds (float): The time in seconds to be converted.
        scaled_data (numpy array, optional): An array of values to be scaled according to the time conversion.

    Returns:
        tuple: (converted time value, unit name, optionally scaled data array)
    """

    time_units = [
        ("sec", 1),
        ("min", 60),
        ("hour", 60),
        ("day", 24),
        ("year", 365),
    ]

    unit_index = 0
    converted_value = seconds
    final_value = seconds
    final_unit = "sec"

    while converted_value >= 1.0 and unit_index < len(time_units) - 1:
        converted_value /= time_units[unit_index + 1][1]

        if converted_value >= 1.0:
            unit_index += 1
            final_value = converted_value
            final_unit = time_units[unit_index][0]

            if scaled_data is not None:
                scaled_data /= time_units[unit_index + 1][1]

    if scaled_data is not None:
        return final_value, final_unit, scaled_data

    return final_value, final_unit
