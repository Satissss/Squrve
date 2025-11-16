import pandas as pd
import math


def slice_schema_df(schema_df: pd.DataFrame, slice_size: int = 200, slice_num: int = 10):
    if not isinstance(schema_df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    total_rows = len(schema_df)

    if total_rows == 0:
        return [schema_df]

    result = []

    # Priority 1: slice_size
    if isinstance(slice_size, int) and slice_size > 0:
        if slice_size >= total_rows:
            return [schema_df]  # 返回自身
        else:
            for start in range(0, total_rows, slice_size):
                result.append(schema_df.iloc[start:start + slice_size])
            return result

    # Priority 2: slice_num
    if isinstance(slice_num, int) and slice_num > 0:
        if slice_num <= 1:  # 修正：当要求1个或更少分片时返回整个DataFrame
            return [schema_df]

        size = math.ceil(total_rows / slice_num)
        for start in range(0, total_rows, size):
            result.append(schema_df.iloc[start:start + size])
        return result

    return [schema_df]
