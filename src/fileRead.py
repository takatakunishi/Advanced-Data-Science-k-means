import sys, os, re
import csvRead

sys.path.append(os.pardir)

def readData():
  return csvRead.readCSV(
    f"../file/sangyohi.csv"
  )