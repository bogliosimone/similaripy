from  .cython_code.s_plus import s_plus

FORMAT_OUTPUT='csr'
VERBOSE = True
K = 100
THRESHOLD = 0
TARGET_ROWS = None # compute all the rows

def dot_product(matrix1, matrix2,
    k=K, threshold=THRESHOLD,
    target_rows=TARGET_ROWS,
    verbose=VERBOSE,
    format_output=FORMAT_OUTPUT
    ):
    return s_plus(
        matrix1, matrix2=matrix2,
        k=k, threshold=threshold,
        target_rows=target_rows,
        verbose=verbose,
        format_output=format_output) 