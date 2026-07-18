import sys
try:
    import matplotlib
    print("matplotlib", matplotlib.__version__)
except Exception as e:
    print("matplotlib MISSING:", e)
try:
    import duckdb
    print("duckdb ok")
except Exception as e:
    print("duckdb MISSING:", e)
sys.stdout.flush()