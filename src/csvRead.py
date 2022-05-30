import pandas as pd
import sys, os

sys.path.append(os.pardir)

# clumnNamesはタイムスタンプが格納されているカラム名を指定
def readCSV(fileName,):
  s = pd.read_csv(
    fileName,
    encoding="UTF8",
    header=None
  )
  return s


def readCSVtoDict(fileName):
  s = pd.read_csv(fileName, header=None, index_col=0, squeeze=True).to_dict()
  return s

def saveCSV(data, fileName):
  data.to_csv(fileName, mode='w')

def loadCSV(fileName):
  s = pd.read_csv(
      fileName,
      encoding="UTF8",
      index_col="time",
      date_parser=lambda x: pd.to_datetime(x),
    )
  return s