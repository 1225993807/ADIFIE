def normalize_column(column, MISSING_DATA):
    integers = []
    for element in column:
        if element != MISSING_DATA:
            integers.append(float(element))

    min_value = min(integers)
    max_value = max(integers)

    normalized_column = []
    for element in column:
        if element != MISSING_DATA:
            if max_value - min_value == 0:
                normalized_element = round((float(element) - min_value) / 1, 6)
            else:
                normalized_element = round((float(element) - min_value) / (max_value - min_value), 6)
            normalized_column.append(normalized_element)
        else:
            normalized_column.append(element)

    return normalized_column
