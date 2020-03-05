import re


# ------------------------------------------------------------------------------------------------ #
def ConvertFileNameToSeconds(fileName):
	import re
	import os
	
	baseName = os.path.basename(fileName)
	
	timeStampRegex = re.compile("(\w*)\-(\d+)")
	timeStampSearch = timeStampRegex.search(baseName)

	if timeStampSearch != None:
		timeStamp = timeStampSearch.group(2)
	else:
		ex = TimeStampFailure('Time Stamp Failure')
		raise ex
	
	
	timeInSeconds = FileNameTimeStampToSecond(fileNameTimeStamp)
	
	return timeInSeconds
# ------------------------------------------------------------------------------------------------ #




# ------------------------------------------------------------------------------------------------ #
class TimeStampFailure(Exception):
	pass
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
class FileExtensionFailure(Exception):
	pass
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
class SelfVectorFailure(Exception):
	pass
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
class ExperimentCodeFailure(Exception):
	pass
# ------------------------------------------------------------------------------------------------ #



# ------------------------------------------------------------------------------------------------ #
def ProcessTimeStamp(fileName):
	import re
	import os
	
	baseName = os.path.basename(fileName)
	
	timeStampRegex = re.compile("(\w*)\-(\d+)")
	timeStampSearch = timeStampRegex.search(baseName)

	if timeStampSearch != None:
		timeStamp = timeStampSearch.group(2)
	else:
		ex = TimeStampFailure('Time Stamp Failure')
		raise ex
		
	return timeStamp
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
def ProcessFileNameExtension(fileName):
	import re
	import os
	
	baseName = os.path.basename(fileName)
	
	extRegex = re.compile("\.(\w+)")
	extSearch = extRegex.search(baseName)

	if extSearch != None:
		extension = extSearch.group(1)
	else:
		ex = FileExtensionFailure('File Extension Failure')
		raise ex
		
	return extension
# ------------------------------------------------------------------------------------------------ #





# ------------------------------------------------------------------------------------------------ #
def ensure_dir(f):
    import os
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def RenameAndCopyImageFile(destBaseDir, fileName, diagnostics=False, preflight=False):
	import shutil
	[plateID, timeStamp, fileExtension] = ProcessPlateBarcode(fileName)
	
	newFileName = plateID + '-' + timeStamp + '.' + fileExtension
	
	if diagnostics == True:
		print(fileName + ' --> ' + newFileName)
	
	if preflight == False:
		ensure_dir(destBaseDir + '/' + plateID + '/' + newFileName)
		shutil.copy(fileName, destBaseDir + '/' + plateID + '/' + newFileName)
	
# ------------------------------------------------------------------------------------------------ #
	




# ------------------------------------------------------------------------------------------------ #
def IdentifyRowAndColumnNumber(xCoord, yCoord, xEdge1, yEdge1, dX, dY):
	xCoordR = xCoord - xEdge1
	yCoordR = yCoord - yEdge1
	
	yCoordScaled = yCoordR/dY
	yNumber = round(yCoordScaled, 0)
	
	xCoordScaled = xCoordR/dX
	xNumber = round(xCoordScaled, 0)
	
	xNumberAdj = int(abs(xNumber)+1)
	yNumberAdj = int(abs(yNumber)+1)
	
	return [xNumberAdj, yNumberAdj]
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
RowNumberToLetterConversionDict = {'1':'A', '2':'B', '3':'C', '4':'D', '5':'E', '6':'F', \
'7':'G', '8':'H'}

def ConvertRowAndColumnNumberToWellID(queryRowNumber, queryColumnNumber):
	queryRowLetter = RowNumberToLetterConversionDict[str(queryRowNumber)]

	wellID = queryRowLetter + str(queryColumnNumber)
	return wellID
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
def GenerateCheckSum(text):
	import numpy
	
	textArray = []
	ordArray = []
	subtractionArray = []
	
	for letter in text:
		textArray.append(letter)
		ordArray.append(ord(letter))
		subtractionArray.append(ord(letter) - 55)
		
	j = 0
	checksum = 0
	subtractionArrayCopy = numpy.zeros(8, numpy.int)

	while j < min([len(subtractionArray), 8]):
		subtractionArrayCopy[j] = subtractionArray[j]
		j += 1
	
	j = 0
	while j < min([len(subtractionArray), 8]):
		if subtractionArrayCopy[j] < 10:
			subtractionArrayCopy[j] += 7
		checksum += subtractionArrayCopy[j]
		j += 1
	
	checksumMod = checksum%36
	if checksumMod < 10:
		checksumMod -= 7
	checkcar = chr(checksumMod+55)
	
	#return [textArray, ordArray, subtractionArray, subtractionArrayCopy, checksum, \
	#checksumMod, checkcar]
	
	return checkcar
# ------------------------------------------------------------------------------------------------ #
















# ------------------------------------------------------------------------------------------------ #
def ExportPlateTimeCourses(CollectedWellTimeCourseDict, dirList, baseDir, prefix='SOGRT|CTRL'):
	import pdb
	from numpy import array, float, ones
	from vectorOutput import generateOutputMatrixWithHeaders, writeOutputMatrix

	for dir in dirList:
		WellTimeCourseDict = CollectedWellTimeCourseDict[dir]
		print(dir)
		
		headerList = []
		vectorList = []
		
		for well in gWellList:
			[row, col] = ConvertWellIDToRowAndColumnNumber(well)
			plotNumber = ConvertWellIDToWellNumber(well)
			
			if WellTimeCourseDict[well].hasODData == True:
				opticalDensities = \
				ones(len(WellTimeCourseDict[well].timeValues), float)*WellTimeCourseDict[well].OD
				
			if len(WellTimeCourseDict[well].genotype) > 0:
				genotypes = []
				i = 0
				while i < len(WellTimeCourseDict[well].timeValues):
					genotypes.append(WellTimeCourseDict[well].genotype)
					i += 1
			
			
			headerList.append(well+'_time')
			headerList.append(well+'_r')
			headerList.append(well+'_g')
			headerList.append(well+'_b')
			
			if WellTimeCourseDict[well].hasODData == True:
				headerList.append(well+'_od')
			
			#pdb.set_trace()
			if len(WellTimeCourseDict[well].genotype) > 0:
				headerList.append(well+'_gene')
			
			timesArray = list(array(WellTimeCourseDict[well].timeValues))
			
			vectorList.append(timesArray)
			vectorList.append(list(WellTimeCourseDict[well].meanRed))
			vectorList.append(list(WellTimeCourseDict[well].meanGreen))
			vectorList.append(list(WellTimeCourseDict[well].meanBlue))
			
			if WellTimeCourseDict[well].hasODData == True:
				vectorList.append(list(opticalDensities))
			
			if len(WellTimeCourseDict[well].genotype) > 0:
				vectorList.append(genotypes)
		
		#pdb.set_trace()
		oMatrix = generateOutputMatrixWithHeaders(vectorList, headerList, delimeter=',')
		
		writeOutputMatrix(baseDir+'/'+dir+'/'+dir+'.csv', oMatrix)
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
####################################################################################################
# Functions and global variables
####################################################################################################
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
RowNumberToLetterConversionDict = {'1':'A', '2':'B', '3':'C', '4':'D', '5':'E', '6':'F', \
'7':'G', '8':'H'}

RowLetterToNumberConversionDict = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8}

gRows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
gColumns = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

gWellList = \
['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12',\
 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', \
 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', \
 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12', \
 'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12', \
 'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12', \
 'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12', \
 'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08', 'H09', 'H10', 'H11', 'H12']


queryWellIDRe = re.compile('(\w)(\d*)')

def ConvertRowAndColumnNumberToWellID(queryRowNumber, queryColumnNumber):
	queryRowLetter = RowNumberToLetterConversionDict[str(queryRowNumber)]

	wellID = queryRowLetter + str(queryColumnNumber)
	return wellID

def ConvertWellIDToRowAndColumnNumber(wellID):
	test = queryWellIDRe.search(wellID)
	if test != None:
		queryRow = test.group(1)
		queryColumn = test.group(2)
	
		queryRowNumber = RowLetterToNumberConversionDict[queryRow]
		queryColumnNumber = int(queryColumn)
		
	return [queryRowNumber, queryColumnNumber]
	
def ConvertWellIDToWellNumber(wellID):
	wellNumber = gWellList.index(wellID)
	return wellNumber


def ConvertWellNumberToWellID(wellNumber):
	wellID = gWellList[wellNumber]
	return wellID
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
def ProcessTimeStamp(fileName):
	import re
	import os
	
	baseName = os.path.basename(fileName)
	
	timeStampRegex = re.compile("(\w*)\-(\d+)")
	timeStampSearch = timeStampRegex.search(baseName)

	if timeStampSearch != None:
		timeStamp = timeStampSearch.group(2)
	else:
		ex = TimeStampFailure('Time Stamp Failure')
		raise ex
		
	return timeStamp
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
####################################################################################################
# Vector processing functions
####################################################################################################

# ------------------------------------------------------------------------------------------------ #
def vecLength(x1):
	from numpy import sqrt
	length = sqrt(dotProduct(x1, x1))
	return length
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
def dotProduct(vec1, vec2):
	i = 0
	dotProduct = 0.0
	while i < len(vec1):
		dotProduct = dotProduct + vec1[i]*vec2[i]
		i = i + 1
	return dotProduct
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
def normalize(x1):
	import numpy
	normalized = numpy.zeros(len(x1))
	i = 0
	mag = numpy.sqrt(dotProduct(x1, x1))
	while i < len(x1):
		normalized[i] = x1[i] / mag
		i = i + 1
	
	return normalized
# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #

####################################################################################################





# ------------------------------------------------------------------------------------------------ #
class Well:
	def __init__(self, plateID, wellID):
		self.colorValue = 0
		self.plateID = plateID
		self.wellID = wellID
		self.hasODData = False
		self.occupied = False
		self.OD = 0
		self.colorValues = []
		self.timeValues = []
		self.meanRed = []
		self.meanGreen = []
		self.meanBlue = []
		self.medianRed = []
		self.medianGreen = []
		self.medianBlue = []
		self.opticalDensities = []
		self.genotype = ''
# ------------------------------------------------------------------------------------------------ #





# ------------------------------------------------------------------------------------------------ #
####################################################################################################
# File utilities
####################################################################################################

# ------------------------------------------------------------------------------------------------ #
def ProcessFileNameExtension(fileName):
	import re
	import os
	
	baseName = os.path.basename(fileName)
	[fileName, fileExtension] = os.path.splitext(baseName)
		
	return [fileName, fileExtension]
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
def NormalizeWellIDFormat(wellID):
	import re
	import pdb
	
	queryWellIDRe = re.compile('([A-Ha-h]+)(\d+)')
	
	queryResult = queryWellIDRe.search(wellID)
	
# 	try:
	wellIDNormalized = queryResult.group(1) + queryResult.group(2).zfill(2)
	# except:
# 		pdb.set_trace()
# 		
		
	return wellIDNormalized
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
def NormalizePlateIDFormat(plateID, prefix='SOGRT',zfillNumber=3):
	import re
	import pdb
	
	#pdb.set_trace()
	
	queryPlateIDRe = re.compile('('+prefix+')_*(\d+)')
	
	queryResult = queryPlateIDRe.search(plateID)
	
	plateIDNormalized = queryResult.group(1) + queryResult.group(2).zfill(zfillNumber)
	
	return plateIDNormalized
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
def WriteFittedWellPositions(fileName, dirName, FittedWellPositionsCollectionDict, saveDir):
# Write out the fitted well positions
	from numpy import array, float
	import pdb
	
	#pdb.set_trace()
	
	wellIDs = sorted(FittedWellPositionsCollectionDict[dirName][fileName][2].keys())
	[fileNameBase, fileExtension] = ProcessFileNameExtension(fileName)
	wellCoordsFileName = fileNameBase + '_well_coordinates.csv'
	
	fileHandle = open(saveDir + '/' + wellCoordsFileName, 'w')
	
	data = FittedWellPositionsCollectionDict[dirName][fileName]
	
	for well in wellIDs:
		scale = data[0]
		coord = array(data[2][well], float)/scale
		outputString = NormalizeWellIDFormat(well) + ',' + str(coord[0]) + ',' + str(coord[1]) + '\n'
		fileHandle.write(outputString)
	
	fileHandle.close()
# ------------------------------------------------------------------------------------------------ #





# ------------------------------------------------------------------------------------------------ #
def ExtractPlateIDFromFileName(fileName):
	import re
	fileNameRe = re.compile('(\w+)\-(\d+)\.(\d+)')
	fileNameSearch = fileNameRe.search(fileName)
	plateID = ''
	if fileNameSearch != None:
		plateID = fileNameSearch.group(1)
	return plateID
# ------------------------------------------------------------------------------------------------ #




# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
def ImportPlateODData(plateDataFile, prefix='CTRL|SOGRT'):
	import re
	import pdb
	
	fileHandle = open(plateDataFile, 'r')
	data = fileHandle.readlines()
	fileHandle.close()

	currentPlate = ''
	plateDict = {}

	for line in data:
		print(line.strip())
		dataLine = line.split(',')
		
		#pdb.set_trace()
		
		plate = NormalizePlateIDFormat(dataLine[0], prefix=prefix)
		wellID = dataLine[1]
		row = dataLine[2]
		column = dataLine[3]
		opticalDensity = dataLine[4]
	
		if plate != currentPlate:
			currentPlate = NormalizePlateIDFormat(plate, prefix=prefix)
			plateDict[currentPlate] = {}
	
		normalizedWellID = NormalizeWellIDFormat(wellID)
		#print(currentPlate + ': ' + normalizedWellID + ': ' + opticalDensity
	
		plateDict[currentPlate][normalizedWellID] = float(opticalDensity)
	
	return plateDict
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
def AddODDataToTimeCourses(WellTimeCourseDict, plateODDataDict, plateODDataDictKey, \
prefix='SOGRT|CTRL', occupiedODThreshold=0.2):
	
	from numpy import ones, float
	
	plateODDataDictKey = NormalizePlateIDFormat(plateODDataDictKey, prefix=prefix)
	
	plateODData = plateODDataDict[plateODDataDictKey]
	
	
	for well in gWellList:
		#opticalDensities = ones(len(WellTimeCourseDict[well].timeValues), float)*plateODData[well]
		#WellTimeCourseDict[well].opticalDensities = opticalDensities
		WellTimeCourseDict[well].OD = float(plateODData[well])
		WellTimeCourseDict[well].hasODData = True
		
		if plateODData[well] > occupiedODThreshold:
			WellTimeCourseDict[well].occupied = True
		else:
			WellTimeCourseDict[well].occupied = False
			
		

	return WellTimeCourseDict
# ------------------------------------------------------------------------------------------------ #



# ------------------------------------------------------------------------------------------------ #
def AddODDataToCollectedTimeCourses(CollectedWellTimeCourseDict, plateODDataDict, dirList, \
prefix='SOGRT|CTRL'):
	
	for dir in dirList:
		WellTimeCourseDict = CollectedWellTimeCourseDict[dir]
		
		
		
		WellTimeCourseDict = AddODDataToTimeCourses(WellTimeCourseDict, plateODDataDict, \
		dir, prefix=prefix)

		CollectedWellTimeCourseDict[dir] = WellTimeCourseDict
		
		
	return CollectedWellTimeCourseDict
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
def ImportPlateTimeCourse(fileName, plateID, occupiedODThreshold=0.2):
	
	import pdb
	from numpy import array, float
	
	dataHandle = open(fileName, 'r')
	data = dataHandle.readlines()
	dataHandle.close()
	
	
	WellTimeCourseDict = {}
	
	lines = []
	
	for line in data:
		line = line.strip()
		line = line.split(',')
		lines.append(line)
	
	keys = lines[0]
	lines.pop(0)
	
	data = array(lines, float)
	
	i = 0
	dataDict = {}
	while i < data.shape[1]:
		dataDict[keys[i]] = data[:,i]
		i += 1
	
	
	wells = []
	for key in keys:
		well = key.split('_')[0]
		if well not in wells:
			if well != '':
				wells.append(well)
	
	
	for well in wells:
		WellTimeCourseDict[well] = Well(plateID, well)
		WellTimeCourseDict[well].timeValues = dataDict[well+'_time']
		WellTimeCourseDict[well].meanRed = dataDict[well+'_r']
		WellTimeCourseDict[well].meanGreen = dataDict[well+'_g']
		WellTimeCourseDict[well].meanBlue = dataDict[well+'_b']
		
		if dataDict.has_key(well+'_od'):
			WellTimeCourseDict[well].OD = dataDict[well+'_od'][0]
			WellTimeCourseDict[well].hasODData = True
			if WellTimeCourseDict[well].OD > occupiedODThreshold:
				WellTimeCourseDict[well].occupied = True
			else:
				WellTimeCourseDict[well].occupied = False
		else:
			WellTimeCourseDict[well].hasODData = False
		
		#pdb.set_trace()
	
	return WellTimeCourseDict
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
def FindMaxTimeValueInWellTimeCourseDict(WellTimeCourseDict):
	
	import pdb
	
	maxTimeValue = 0
	
	for well in gWellList:
		maxWellTime = max(WellTimeCourseDict[well].timeValues)
		if maxWellTime > maxTimeValue:
			maxTimeValue = maxWellTime
		
	return maxTimeValue
# ------------------------------------------------------------------------------------------------ #





# ------------------------------------------------------------------------------------------------ #
def FindColoredRegistrationMark(scaledImg, registrationColor, colorThreshold=30, dog_max_sigma=10, \
dog_threshold=10, dog_expectedArea=4):
	
	import scipy.spatial.distance as spsd
	from numpy import array, argsort, float, zeros_like
	from skimage.feature import blob_dog
	import pdb
	
	# Do a color distance transformation
	pixels = scaledImg.reshape(-1,3)
	colorDistances = spsd.cdist(pixels, [registrationColor])
	colorDistances *= (255.0/colorDistances.max())
	colorDistanceImg = colorDistances.reshape(scaledImg.shape[0], scaledImg.shape[1])
	
	colorThresholdedImg = zeros_like(colorDistanceImg)
	colorThresholdedImg[colorDistanceImg > colorThreshold] = 0
	colorThresholdedImg[colorDistanceImg <= colorThreshold] = 255
	
	# Just a reminder, scikit works in a y-x coordinate system, but I like to work in an x-y system
	# so you will see some indices flipped. 
	
	try:
		blobs_dog = blob_dog(colorThresholdedImg, max_sigma=dog_max_sigma, threshold=dog_threshold)
		
		if len(blobs_dog) > 0:
			diffsToExpectedArea = []
			i = 0
			while i < len(blobs_dog):
				diffsToExpectedArea.append(abs(dog_expectedArea - blobs_dog[i][2]))
				i += 1
	
			bestIndex = argsort(diffsToExpectedArea)[0]
			registrationMarkCoord = array([blobs_dog[bestIndex][1], blobs_dog[bestIndex][0]], float)
			registrationMarkFound = True
		else:
			registrationMarkCoord = array([0., 0.], float)
			registrationMarkFound = False
	except:
		registrationMarkCoord = array([0., 0.], float)
		registrationMarkFound = False
		
	return registrationMarkCoord, registrationMarkFound
# ------------------------------------------------------------------------------------------------ #



# ------------------------------------------------------------------------------------------------ #
# Find the pink and blue registration marks
def FindPlateOrientationFromTopBarcodeRegistrationMarks(scaledImg, \
blueRegistrationColor, pinkRegistrationColor, blueOriginInit, pinkOriginInit, diagnostics=True):

	import numpy
	from numpy import array, arccos, arcsin, arctan
	import pdb
	
	defaultBlueToPinkVector = pinkOriginInit - blueOriginInit
	
	blueCoord, blueMarkFound = FindColoredRegistrationMark(scaledImg, blueRegistrationColor)
	pinkCoord, pinkMarkFound = FindColoredRegistrationMark(scaledImg, pinkRegistrationColor)	
	
	# If we can't locate either the pink or blue registration mark
	if pinkMarkFound == False and blueMarkFound == False:
		originGuess = True
		thetaGuess = True
		blueCoord = blueOriginInit
		pinkCoord = pinkOriginInit

	# Can find the blue blob, but not the pink blob
	elif pinkMarkFound == False and blueMarkFound == True:
		originGuess = False
		thetaGuess = True	
		pinkCoord = blueCoord + defaultBlueToPinkVector
	
	# Can't find blue blob but can find pink mark
	elif pinkMarkFound == True and blueMarkFound == False:
		originGuess = True
		thetaGuess = True
		blueCoord = pinkCoord - defaultBlueToPinkVector
	
	elif pinkMarkFound == True and blueMarkFound == True:
		originGuess = False
		thetaGuess = False
	
	bluePinkVector = pinkCoord - blueCoord
	registrationRulerLength = vecLength(bluePinkVector)
	# nBluePinkVector = normalize(bluePinkVector)
# 	theta = arccos(nBluePinkVector[0])

	sinTheta = bluePinkVector[1]/vecLength(bluePinkVector)
	
# 	tanTheta = bluePinkVector[1]/bluePinkVector[0]
	theta = arcsin(sinTheta)
	
	
# 	pdb.set_trace()

	
	return [blueCoord, pinkCoord, theta, registrationRulerLength, originGuess, thetaGuess]
# ------------------------------------------------------------------------------------------------ #



# ------------------------------------------------------------------------------------------------ #
def FindRingsOn96WellPlateImage(scaledImg, houghRadii):

	from skimage import data, color
	from numpy import zeros_like, arange, argsort
	from skimage.feature import peak_local_max, canny
	from skimage.transform import hough_circle
	
	image_gray = color.rgb2gray(scaledImg)

	blackEdges = zeros_like(image_gray)
	blackEdges[image_gray > 0.6] = 1
	blackEdges[image_gray <= 0.6] = 0
	edges = canny(blackEdges)
	
	houghRes = hough_circle(edges, houghRadii)

	centers = []
	accums = []
	radii = []

	numPeaks = 96

	for radius, h in zip(houghRadii, houghRes):
		# For each radius, extract 96 circles
		peaks = peak_local_max(h, num_peaks=numPeaks)
		centers.extend(peaks)
		accums.extend(h[peaks[:, 0], peaks[:, 1]])
		radii.extend([radius]*numPeaks)

	i = 0
	topAccumIndices = argsort(accums)[::-1][:96]
	centersToReturn = []
	
	for index in topAccumIndices:
		centerY, centerX = centers[index]
		radius = radii[index]
		centersToReturn.append([centerX, centerY, radius])
	
	return centersToReturn
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
def DrawRingsOnPhotograph(img, ringCoordinates, ringColor):
	
	from skimage.draw import circle_perimeter
	
	for ring in ringCoordinates:
		cy, cx = circle_perimeter(int(ring[1]), int(ring[0]), int(ring[2]))
		img[cy, cx] = ringColor
	
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
def DrawRegistrationMarkOnPhotograph(img, markCoord, markRadius, markColor):
	
	from skimage.draw import circle_perimeter
	import pdb
	
	try:
		cy, cx = circle_perimeter(int(markCoord[1]), int(markCoord[0]), int(markRadius))
	except:
		pdb.set_trace()
		
	img[cy, cx] = markColor
	
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
def Calculate96WellPositions(originCoord, theta, registrationRuler, \
wellSepXInitRatio, wellSepYInitRatio, originToA12InitRatio):
	
	import pdb
		
	calculatedWellPositionsDict = {}
	
	for well in gWellList:
		calculatedWellPositionsDict[well] = \
		CalculatePositionOfWellOn96WellPlate(well, originCoord, theta, registrationRuler, \
		wellSepXInitRatio, wellSepYInitRatio, originToA12InitRatio)
	
	return calculatedWellPositionsDict
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
def CalculatePositionOfWellOn96WellPlate(well, originCoord, theta, registrationRuler, \
wellSepXInitRatio, wellSepYInitRatio, originToA12InitRatio):

	from numpy import array, cos, sin
	import pdb

	# Remember, theta is the direction of the blue to pink vector, so really defines the y axis
	# in the coordinate system
	# We are defining the plate vectors. p1 is the x axis of the plate, running along the rows, 
	# say from A1 to A12. p2 is the y axis, running along the columns from A1 to H1. 


	p1 = array([sin(theta), -1.0*cos(theta)])
	p2 = array([cos(theta), sin(theta)])
	
	wellSepX = wellSepXInitRatio*registrationRuler
	wellSepY = wellSepYInitRatio*registrationRuler
	originToA12 = originToA12InitRatio*registrationRuler

	A1coord = originCoord + originToA12[0]*p1 + originToA12[1]*p2 - 11*wellSepX*p1
	
	[queryRowNumber, queryColumnNumber] = ConvertWellIDToRowAndColumnNumber(well)
	
	wellPosition = originCoord \
	+ originToA12[0]*p1 + originToA12[1]*p2 \
	- wellSepX*(12. - queryColumnNumber)*p1 + wellSepY*(-1. + queryRowNumber)*p2
	
	return wellPosition
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
def AssignPlateImageFeaturesToWells(photoFeatures, WellPositionGuessDict, diagnostics=False):
	
	import pdb
	from numpy import array
	
	blobAssignmentDict = {}
	guessKeys = list(WellPositionGuessDict.keys())

	for blob in photoFeatures:
		coordinates = array([blob[0], blob[1]])
		guessCoordToBlobArray = []
	
		for key in guessKeys:
			guessCoord = WellPositionGuessDict[key]
			guessCoordToBlobArray.append(vecLength(guessCoord - coordinates))
	
		shortestGuessToBlob = min(guessCoordToBlobArray)
		minIndex = guessCoordToBlobArray.index(shortestGuessToBlob)
		
		try:
			bestGuestKey = guessKeys[minIndex]
		except:
			pdb.set_trace()
		
		
		blobAssignmentDict[bestGuestKey] = blob

	return blobAssignmentDict
# ------------------------------------------------------------------------------------------------ #



# ------------------------------------------------------------------------------------------------ #
def FeatureGridResidual(p, featureXCoordArray, featureYCoordArray, originX, originY, \
registrationRulerLength, wellNumbers):
	
	import pdb
	
	[wellSepXRatio, wellSepYRatio, theta, originToA12XRatio, originToA12YRatio] = p
	originToA12Ratio = [originToA12XRatio, originToA12YRatio]
	originCoord = [originX, originY]
	
	residuals = []
	
	i = 0
	while i < len(wellNumbers):
		
		well = wellNumbers[i]
		
		calculatedWellPosition  = CalculatePositionOfWellOn96WellPlate(well, originCoord, \
		theta, registrationRulerLength, wellSepXRatio, wellSepYRatio, originToA12Ratio)
		
# 		pdb.set_trace()
			
		residual = calculatedWellPosition - [featureXCoordArray[i], featureYCoordArray[i]]
		residuals.append(vecLength(residual))
	
		i += 1
		
# 	pdb.set_trace()
	return residuals
# ------------------------------------------------------------------------------------------------ #



# ------------------------------------------------------------------------------------------------ #
def FitFeatureGridCoords(featureAssignmentDict, originCoord, theta, \
registrationRulerLength, wellSepXInitRatio, wellSepYInitRatio, originToA12InitRatio):

	import scipy
	from scipy.optimize import leastsq
	import pdb
	
	wellSepXInit = wellSepXInitRatio*registrationRulerLength
	wellSepYInit = wellSepYInitRatio*registrationRulerLength
	originToA12Init = originToA12InitRatio*registrationRulerLength
	
	# Break out the feature assignment dict into a list of coordinates and a list of well numbers
	featureKeys = list(featureAssignmentDict.keys())
	featureToWellAssignments = []
	coordXArray = []
	coordYArray = []
	
	for key in featureKeys:
		coordX = featureAssignmentDict[key][0]
		coordY = featureAssignmentDict[key][1]
		
		featureToWellAssignments.append(key)
		
		coordXArray.append(coordX)
		coordYArray.append(coordY)
	
	# Fit the feature grid parameters
	
	pInit = [wellSepXInitRatio, wellSepYInitRatio, theta, originToA12InitRatio[0], \
	originToA12InitRatio[1]]
	
# 	pdb.set_trace()
	
	plsq = leastsq(FeatureGridResidual, pInit, \
	args=(coordXArray, coordYArray, originCoord[0], originCoord[1], registrationRulerLength, \
	featureToWellAssignments), maxfev=2000)
	
	fittedWellSepXRatio = plsq[0][0]
	fittedWellSepYRatio = plsq[0][1]
	fittedTheta = plsq[0][2]
	fittedOriginToA12Ratio = [plsq[0][3], plsq[0][4]]
	
	return fittedWellSepXRatio, fittedWellSepYRatio, fittedTheta, fittedOriginToA12Ratio
# ------------------------------------------------------------------------------------------------ #



# ------------------------------------------------------------------------------------------------ #
def FindWellPositionsIn96WellAssayPlateImage(fileName, blueRegistrationColor, \
pinkRegistrationColor, wellSepXInitRatio, wellSepYInitRatio, originToA12InitRatio, \
wellRadiusRatio, blueOriginInit, pinkOriginInit, diagnostics=False, scale=0.2):
	
	import pdb
	import shutil
	from os.path import dirname
	from matplotlib.image import imread
	from matplotlib.pyplot import imshow, show, figure
	from scipy.misc import imresize
	from numpy import arange
	from copy import copy

	enclosingDir = dirname(fileName)
	
	img = imread(fileName)
	scaledImg = imresize(img, scale)

	# Find the plate orientation from the registration marks
	[blueCoord, pinkCoord, theta, registrationRulerLength, originGuess, thetaGuess] = \
	FindPlateOrientationFromTopBarcodeRegistrationMarks(scaledImg, blueRegistrationColor, \
	pinkRegistrationColor, blueOriginInit*scale, pinkOriginInit*scale, \
	diagnostics=diagnostics)
	
	# Guess the positions of the wells based upon the registration marks
	wellPositionGuessDict = \
	Calculate96WellPositions(blueCoord, theta, registrationRulerLength, \
	wellSepXInitRatio, wellSepYInitRatio, originToA12InitRatio)
	
	# Find the condensation rings on the plate image
	houghRadii = arange(wellRadiusRatio*registrationRulerLength*0.8, \
	wellRadiusRatio*registrationRulerLength*1.2, registrationRulerLength*0.02)
	
	rings = FindRingsOn96WellPlateImage(scaledImg, houghRadii)
	
	# Assign the rings to wells	
	ringAssignmentDict = AssignPlateImageFeaturesToWells(rings, wellPositionGuessDict)
	
	if diagnostics == True:
		# Draw the well registration mark positions on the plate
		DrawRegistrationMarkOnPhotograph(scaledImg, blueCoord, 10, (0,255,255))
		DrawRegistrationMarkOnPhotograph(scaledImg, pinkCoord, 10, (255,0,0))
	
		# Draw the rings on the plate
		DrawRingsOnPhotograph(scaledImg, rings, (255,0,0))
	
		# Draw well positions on the plate
		# Still figuring this out, but everything looks OK
		figure()
		imshow(scaledImg)
		show()
	
	# Use the ring positions and assignments to refine the plate orientation	
	fittedWellSepXRatio, fittedWellSepYRatio, fittedTheta, fittedOriginToA12Ratio = \
	FitFeatureGridCoords(ringAssignmentDict, blueCoord, theta, \
	registrationRulerLength, wellSepXInitRatio, wellSepYInitRatio, originToA12InitRatio)
	
	# Calculate the well positions based upon the refined plate parameters
	FittedWellPositionsDict = \
	Calculate96WellPositions(blueCoord, fittedTheta, registrationRulerLength, fittedWellSepXRatio, \
	fittedWellSepYRatio, fittedOriginToA12Ratio)
	
	
	return FittedWellPositionsDict
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
def GetMeanAndMedianColorsInPixelsArray(pixels):
	from numpy import array, mean, median, zeros
	
	if len(pixels) == 0:
		print("Error!")
		meanRed = 0
		meanGreen = 0
		meanBlue = 0
	
		medianRed = 0
		medianGreen = 0
		medianBlue = 0
	else:
		redValueSet = pixels[:,0]
		greenValueSet = pixels[:,1]
		blueValueSet = pixels[:,2]

		meanRed = mean(redValueSet)
		meanGreen = mean(greenValueSet)
		meanBlue = mean(blueValueSet)
	
		medianRed = median(redValueSet)
		medianGreen = median(greenValueSet)
		medianBlue = median(blueValueSet)
	
	
	return [[meanRed, meanGreen, meanBlue], [medianRed, medianGreen, medianBlue]]
# ------------------------------------------------------------------------------------------------ #



# ------------------------------------------------------------------------------------------------ #
def MeasureAverageWellColorsIn96WellPlate(fullFileName, FittedWellPositionsDict, wellRadius):

# Note: modified this function on 2015-11-13 to remove bug that incorrectly calculated 
# colors if fitted well positions were supplied in a format appropriately scaled for the original
# unscaled input image

	import pdb
	from matplotlib.image import imread
	from scipy.misc import imresize
	from skimage.draw import circle
	
	
	img = imread(fullFileName)
	
	wellKeys = sorted(FittedWellPositionsDict.keys())

	WellAverageColorsDict = {}
	
	for well in wellKeys:
		coordinates = FittedWellPositionsDict[well]
		coordX = int(coordinates[0])
		coordY = int(coordinates[1])
		circleRadius = int(0.5*wellRadius)
				
		rr, cc = circle(coordY, coordX, circleRadius)
		
		pixels = img[rr, cc]
		
		meanAndMedianColors = GetMeanAndMedianColorsInPixelsArray(pixels)
		
		WellAverageColorsDict[well] = meanAndMedianColors
	
	return WellAverageColorsDict
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
def Calculate96WellPositionsOnStoragePlate(A1Cooord, theta, wellSepX, wellSepY):

# Calculates the position of all wells on a 96-well storage plate. 
# This algorithm isn't that general purpose and assumes a fairly well aligned plate, of a 
# consistent scale (with all other images in the series) and with A1 closest to the image origin 
# (top left). 


	import pdb

	calculatedWellPositionsDict = {}
	
	for well in gWellList:
		calculatedWellPositionsDict[well] = \
		CalculatePositionOfWellOn96WellStoragePlate(well, A1Cooord, theta, wellSepX, wellSepY)
	
	return calculatedWellPositionsDict
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
def CalculatePositionOfWellOn96WellStoragePlate(well, A1Coord, theta, wellSepX, wellSepY):

# Calculates the position of a specified well on a 96-well storage plate. 
# This algorithm isn't that general purpose and assumes a fairly well aligned plate, of a 
# consistent scale (with all other images in the series) and with A1 closest to the image origin 
# (top left). 

	from numpy import array, cos, sin
	import pdb

	p1 = array([sin(theta), -1.0*cos(theta)])
	p2 = array([cos(theta), sin(theta)])
	
	[queryRowNumber, queryColumnNumber] = ConvertWellIDToRowAndColumnNumber(well)
	
	wellPosition = A1Coord + wellSepX*(queryColumnNumber-1)*p1 + wellSepY*(queryRowNumber-1)*p2
	
	return wellPosition
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
def FindRingsOn96WellStoragePlateImage(scaledImg, houghRadii, diagnostics=False):
# Adaptation of FindRingsOn96WellPlateImage that looks for lighter rings in image of storage plate

	from skimage import data, color
	from numpy import zeros_like, arange, argsort
	from skimage.feature import peak_local_max, canny
	from skimage.transform import hough_circle
	import pdb
	from matplotlib.pyplot import imshow, show, figure
	from skimage.filters import threshold_otsu, threshold_adaptive
	
	image_gray = color.rgb2gray(scaledImg)

	blackEdges = zeros_like(image_gray)
	cutoff = 0.8
	blackEdges[image_gray > cutoff] = 1
	blackEdges[image_gray <= cutoff] = 0
	edges = canny(blackEdges)
	
	
	if diagnostics:
		figure()
		imshow(image_gray)
		figure()
		imshow(blackEdges)
		show()
	
	houghRes = hough_circle(edges, houghRadii)

	centers = []
	accums = []
	radii = []

	numPeaks = 96

	for radius, h in zip(houghRadii, houghRes):
		# For each radius, extract 96 circles
		peaks = peak_local_max(h, num_peaks=numPeaks)
		centers.extend(peaks)
		accums.extend(h[peaks[:, 0], peaks[:, 1]])
		radii.extend([radius]*numPeaks)

	i = 0
	topAccumIndices = argsort(accums)[::-1][:96]
	centersToReturn = []
	
	for index in topAccumIndices:
		centerY, centerX = centers[index]
		radius = radii[index]
		centersToReturn.append([centerX, centerY, radius])
	
	return centersToReturn
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
def FeatureGridFor96WellStoragePlateResidual(p, featureXCoordArray, featureYCoordArray, \
wellNumbers):
	
	import pdb
	
	[A1CoordX, A1CoordY, wellSepX, wellSepY, theta] = p
	
	A1Coord = [A1CoordX, A1CoordY]
	
	residuals = []
	
	i = 0
	while i < len(wellNumbers):
		
		well = wellNumbers[i]
		
		calculatedWellPosition  = \
		CalculatePositionOfWellOn96WellStoragePlate(well, A1Coord, theta, wellSepX, wellSepY)
		
		residual = calculatedWellPosition - [featureXCoordArray[i], featureYCoordArray[i]]
		residuals.append(vecLength(residual))
	
		i += 1

	return residuals
# ------------------------------------------------------------------------------------------------ #




# ------------------------------------------------------------------------------------------------ #
def Refine96WellStoragePlateCoords(featureAssignmentDict, A1CoordInit, wellSepXInit, wellSepYInit, \
thetaInit):

	import scipy
	from scipy.optimize import leastsq
	import pdb
	
	# Break out the feature assignment dict into a list of coordinates and a list of well numbers
	featureKeys = list(featureAssignmentDict.keys())
	featureToWellAssignments = []
	coordXArray = []
	coordYArray = []

	A1CoordXInit = A1CoordInit[0]
	A1CoordYInit = A1CoordInit[1]
	
	for key in featureKeys:
		coordX = featureAssignmentDict[key][0]
		coordY = featureAssignmentDict[key][1]
		
		featureToWellAssignments.append(key)
		
		coordXArray.append(coordX)
		coordYArray.append(coordY)
	
	# Fit the feature grid parameters
	
	pInit = [A1CoordXInit, A1CoordYInit, wellSepXInit, wellSepYInit, thetaInit]
	
# 	pdb.set_trace()
	
	plsq = leastsq(FeatureGridFor96WellStoragePlateResidual, pInit, \
	args=(coordXArray, coordYArray, featureToWellAssignments), maxfev=2000)
	
	A1CoordRef = [plsq[0][0], plsq[0][1]]
	wellSepXRef = plsq[0][2]
	wellSepYRef = plsq[0][3]
	thetaRef = plsq[0][4]
	
	return A1CoordRef, wellSepXRef, wellSepYRef, thetaRef
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
def EstimateWellGridParameters(rings, A1CoordXInit, A1CoordYInit, wellSepXInit, wellSepYInit, \
scale):
	
	import operator
	from numpy import mean, array, float
	
	ringsSortedByX = sorted(rings, key=operator.itemgetter(0))
	ringsSortedByY = sorted(rings, key=operator.itemgetter(1))
	
	# Use the ring data to guess the position of the well grid
	minRingX = ringsSortedByX[0][0]
	maxRingX = ringsSortedByX[-1][0]
	minRingY = ringsSortedByY[0][1]
	maxRingY = ringsSortedByY[-1][1]
	
	columnSepsSeen = round((maxRingX - minRingX)/(wellSepXInit*scale))
	rowSepsSeen = round((maxRingY - minRingY)/(wellSepXInit*scale))
	
	if rowSepsSeen < 7.0 and minRingX > A1CoordXInit*scale:
		A1CoordXGuess = A1CoordXInit*scale
	elif minRingX < A1CoordXInit*scale:
		A1CoordXGuess = minRingX
	else:
		A1CoordXGuess = A1CoordXInit*scale
	
	if columnSepsSeen < 11.0 and minRingY > A1CoordYInit*scale:
		A1CoordYGuess = A1CoordYInit*scale
	elif minRingY < A1CoordYInit*scale:
		A1CoordYGuess = minRingY
	
	A1CoordGuess = array([A1CoordXGuess, A1CoordYGuess], float)
	wellSepXGuess = wellSepXInit*scale
	wellSepYGuess = wellSepYInit*scale
	
	
	ringRadii = []
	for ring in rings:
		ringRadii.append(ring[2])
	
	meanRingRadius = mean(ringRadii)/scale


	return [A1CoordGuess, wellSepXGuess, wellSepYGuess, meanRingRadius]

# ------------------------------------------------------------------------------------------------ #





# ------------------------------------------------------------------------------------------------ #
def FindWellPositionsIn96WellStoragePlateImage(fileName, A1CoordXInit, A1CoordYInit, \
wellSepXInit, wellSepYInit, wellRadiusInit, thetaInit, diagnostics=False, scale=0.2):

# Function to find well positions in image of 96 well polypropylene storage plate
# Photo is normally taken from the bottom, but algorithm assumes that the image has been manually
# flipped by human operator. Assumes that plate is in correct orientation, with A1 at the top
# right of the flipped image, or is the closest well to the image coordinate system. 

	import pdb
	from os.path import dirname
	from matplotlib.image import imread
	from matplotlib.pyplot import imshow, show, figure
	from scipy.misc import imresize
	from numpy import arange, array, float

	enclosingDir = dirname(fileName)
	
	img = imread(fileName)
	scaledImg = imresize(img, scale)

	# There is no registration mark on the image (except for the barcode, which at the time of 
	# writing this algorithm, I can't easily read or even identify). 
	
	# Make a guess at the position of the wells. 
	# This algorithm isn't fantastically general purpose. It assumes that the plate is in 
	# approximately the same position in each photograph it processes, at that it is approximately
	# straight, and with approximately the same scale. 
	

	# Find the condensation rings on the plate image
	houghRadii = arange(wellRadiusInit*scale*0.8, wellRadiusInit*scale*1.2, \
	wellRadiusInit*scale*0.05)
	
	rings = FindRingsOn96WellStoragePlateImage(scaledImg, houghRadii, diagnostics=False)
	
	[A1CoordGuess, wellSepXGuess, wellSepYGuess, meanRingRadius] = \
	EstimateWellGridParameters(rings, A1CoordXInit, A1CoordYInit, wellSepXInit, wellSepYInit, scale)
	
	
	wellPositionGuessDict = \
	Calculate96WellPositionsOnStoragePlate(A1CoordGuess, thetaInit, wellSepXGuess, wellSepYGuess)
	
		
	# Assign the rings to wells	
	ringAssignmentDict = AssignPlateImageFeaturesToWells(rings, wellPositionGuessDict)
		
	# Use the ring positions and assignments to refine the plate orientation	
	A1CoordRef, wellSepXRef, wellSepYRef, thetaRef = \
	Refine96WellStoragePlateCoords(ringAssignmentDict, A1CoordGuess*scale, \
	wellSepXGuess*scale, wellSepYGuess*scale, thetaInit)
	
	# Calculate the well positions based upon the refined plate parameters
	# Remember, these are for the full sized image, not the scaled one!
	fittedWellPositionsDict = \
	Calculate96WellPositionsOnStoragePlate(array(A1CoordRef, float)/scale, thetaRef, \
	wellSepXRef/scale, wellSepYRef/scale)
	
	scaledFittedWellPositionsDict = \
	Calculate96WellPositionsOnStoragePlate(A1CoordRef, thetaRef, wellSepXRef, \
	wellSepYRef)
	
	if diagnostics == True:
		figure()
		# Draw the rings on the plate
		DrawRingsOnPhotograph(scaledImg, rings, (255,0,0))
		
		# Draw the fitted well positions
		DrawFittedWellPositionsOnPhotograph(scaledImg, scaledFittedWellPositionsDict, \
		meanRingRadius*scale, (0,255,0))
		
		imshow(scaledImg)
		show()
# 		pdb.set_trace()
 		
	return [fittedWellPositionsDict, meanRingRadius, scaledImg]
# ------------------------------------------------------------------------------------------------ #





# ------------------------------------------------------------------------------------------------ #
def DrawFittedWellPositionsOnPhotograph(img, fittedWellPositionsDict, markRadius, markColor):
	
	from skimage.draw import circle_perimeter
	import pdb
	
	wellKeys = fittedWellPositionsDict.keys()
	
	for well in wellKeys:
		markCoord = fittedWellPositionsDict[well]
	
		try:
			cy, cx = circle_perimeter(int(markCoord[1]), int(markCoord[0]), int(markRadius))
		except:
			pdb.set_trace()
		
		img[cy, cx] = markColor
	
# ------------------------------------------------------------------------------------------------ #



# ------------------------------------------------------------------------------------------------ #
def GetMeanValueInPixelsArray(pixels):
	from numpy import array, mean, median, zeros
	
	meanValue = mean(pixels)
# 	medianValue = median(pixels)

	return meanValue
# ------------------------------------------------------------------------------------------------ #



# ------------------------------------------------------------------------------------------------ #
def MeasureWellsIn96WellPlateImage(fullFileName, fittedWellPositionsDict, ringRadius, \
wellRadiusScaleFactor=0.8, diagnostics=False):

# Note: modified this function on 2015-11-13 to remove bug that incorrectly calculated 
# colors if fitted well positions were supplied in a format appropriately scaled for the original
# unscaled input image

	import pdb
	from matplotlib.image import imread
	from scipy.misc import imresize
	from skimage.draw import circle
	import scipy.spatial.distance as spsd
	from skimage import data, color
	from skimage.filters import threshold_otsu, threshold_adaptive
	import pylab
	from skimage.exposure import adjust_sigmoid, equalize_adapthist
	
	
	img = imread(fullFileName)
		
	imgContrastAdj = equalize_adapthist(img)
	
	imgForMeasurement = color.rgb2gray(imgContrastAdj)
	
	if diagnostics==True:
		figure()
		imshow(imgForMeasurement, cmap=pylab.gray())
		show()
	
	wellKeys = sorted(fittedWellPositionsDict.keys())

	wellAverageValuesDict = {}
	
	for well in wellKeys:
		coordinates = fittedWellPositionsDict[well]
		coordX = int(coordinates[0])
		coordY = int(coordinates[1])
		circleRadius = int(wellRadiusScaleFactor*ringRadius)
				
		rr, cc = circle(coordY, coordX, circleRadius)
		
		pixels = imgForMeasurement[rr, cc]
		
		meanValue = GetMeanValueInPixelsArray(pixels)
		
		wellAverageValuesDict[well] = meanValue
	
	return [wellAverageValuesDict, imgForMeasurement]
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
def AssignOccupancyToWells(wellAverageValueDict, threshold=0.7):

	wellKeys = wellAverageValueDict
	wellOccupancyDict = {}
	
	for well in wellKeys:
		if wellAverageValueDict[well] > threshold:
			wellOccupancyDict[well] = False
		else:
			wellOccupancyDict[well] = True
	
	return wellOccupancyDict
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
def ImportSudokuCatalogFromCSV(fileName, rowPools=None, colPools=None):
# Code for importing Sudoku library catalog from a CSV file and outputting to a SudokuGrid like
# object.
# Assumes that catalog is in the format: plate, well, transposon coordinate, disrupted gene name.

	import operator
	import pdb
	
	fileHandle = open(fileName, 'r')
	data = fileHandle.readlines()
	
	lineData = []
	
	for line in data:
		if line[0] != '#':
			lineData.append(line.strip().split(','))
	
	# Sort the lines by plate
	lineData = sorted(lineData, key=operator.itemgetter(1))
	
	# Initialize a plate dict that is indexed by plate number and holds a dict of wells
	
	plateDict = {}
	
	# Go through the lines in the catalog and add them to the plate dict
	
	for line in lineData:
		if line[0] not in plateDict.keys():
			plateDict[line[0]] = {}
		
		try:
			plateDict[line[0]][NormalizeWellIDFormat(line[1])] = [int(line[2]), line[3]]
		except:
			pdb.set_trace()
	
	for plate in sorted(plateDict.keys()):
		filledWells = plateDict[plate].keys()
		
		for well in gWellList:
			if well not in filledWells:
				plateDict[plate][well] = [-1, 'Blank']
# 				print("Plate " + plate + " well " + well + " is blank.")
	
		
	return plateDict
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
def TestOccupancyAgainstCatalog(wellOccupancyDict, wellContentsDict):
	

	occupancyTestDict = {}
	occupancyTestSummaryDict = {'CC':[], 'DO':[]}
	
	for well in gWellList:
		if wellOccupancyDict[well] == True and wellContentsDict[well][0] > -1:
			occupancyTestDict[well] = 'OK'
		elif wellOccupancyDict[well] == False and wellContentsDict[well][0] > -1:
			occupancyTestDict[well] = 'DO'
			occupancyTestSummaryDict['DO'].append(well)
		elif wellOccupancyDict[well] == True and wellContentsDict[well][0] == -1:
			occupancyTestDict[well] = 'CC'
			occupancyTestSummaryDict['CC'].append(well)
		elif wellOccupancyDict[well] == False and wellContentsDict[well][0] == -1:
			occupancyTestDict[well] = 'OK'
	
	return [occupancyTestDict, occupancyTestSummaryDict]

# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
def MarkUpPlateImageWithOccupancy(measurementImg, wellOccupancyDict, fittedWellPositionsDict, \
radius, scale=0.2, occupiedColor=(255,0,0), unoccupiedColor=(0,0,255)):
	
	import pdb
	from matplotlib.pyplot import imshow, show, figure
	
	from skimage.color import gray2rgb
	from copy import copy
	from scipy.misc import imresize

	
	markupImg = gray2rgb(copy(imresize(measurementImg, scale)))
	
	
	wellsWithOccupancyData = wellOccupancyDict.keys()
	
	for well in wellsWithOccupancyData:
		
		fittedWellPosition = {well:fittedWellPositionsDict[well]*scale}
	
		if wellOccupancyDict[well] == True:
			DrawFittedWellPositionsOnPhotograph(markupImg, fittedWellPosition, \
			radius*scale, occupiedColor)
		else:
			DrawFittedWellPositionsOnPhotograph(markupImg, fittedWellPosition, \
			radius*scale, unoccupiedColor)
		
	figure()
	imshow(markupImg)
	show()
	
	return
# ------------------------------------------------------------------------------------------------ #




# ------------------------------------------------------------------------------------------------ #
def MarkUpPlateImageWithOccupancyTest(measurementImg, wellOccupancyDict, occupancyTestDict, \
fittedWellPositionsDict, radius, figTitle=None, markedUpFileName=None, scale=0.2, \
occupiedColor=(255,0,0), unoccupiedColor=(0,0,255)):
	
	import pdb
	from matplotlib.pyplot import imshow, show, figure, annotate, imsave, savefig, title
	
	from skimage.color import gray2rgb
	from copy import copy
	from scipy.misc import imresize

	
	markupImg = gray2rgb(copy(imresize(measurementImg, scale)))
	
	
	wellsWithOccupancyData = wellOccupancyDict.keys()
	
	for well in wellsWithOccupancyData:
		
		fittedWellPosition = {well:fittedWellPositionsDict[well]*scale}
	
		if wellOccupancyDict[well] == True:
			DrawFittedWellPositionsOnPhotograph(markupImg, fittedWellPosition, \
			radius*scale, occupiedColor)
		else:
			DrawFittedWellPositionsOnPhotograph(markupImg, fittedWellPosition, \
			radius*scale, unoccupiedColor)
			
	
	figure()	
	imshow(markupImg)
	
	for well in wellsWithOccupancyData:
		if occupancyTestDict[well] == 'CC':
			fittedWellPosition = fittedWellPositionsDict[well]*scale
			try:
				coords = (int(fittedWellPosition[0]), int(fittedWellPosition[1]))
			except:
				pdb.set_trace()
			
			annotate(s='CC', xy=coords)
		elif occupancyTestDict[well] == 'DO':
			fittedWellPosition = fittedWellPositionsDict[well]*scale
			coords = (int(fittedWellPosition[0]), int(fittedWellPosition[1]))
			annotate(s='DO', xy=coords)
	
	if figTitle!= None:
		title(figTitle)
	
	if markedUpFileName != None:
		savefig(markedUpFileName)
	
	show()
	
	return
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
def WriteUpdatedSudokuCatalog(updatedCatalogFileName, updatedCatalog):
# Writes out an updated sudoku catalog from image analysis of storage plate images
	import pdb
	
	fileHandle = open(updatedCatalogFileName, 'w')
	
	plateKeys = updatedCatalog.keys()
	
	pdb.set_trace()
	
	headerStr = '#Plate,Well,Transposon Coord,Feature,Average Value,Occupancy,Occupancy Test\n'
	
	fileHandle.write(headerStr)
	
	for plateKey in plateKeys:
		
		for well in gWellList:
			
			catalogEntry = updatedCatalog[plateKey][well]
			
# 			pdb.set_trace()
			
			transposonCoord = str(catalogEntry[0])
			feature = str(catalogEntry[1])
			averageValue = str(catalogEntry[2])
			occupancy = str(catalogEntry[3])
			occupancyTest = str(catalogEntry[4])
			
			outputStr = plateKey + ',' + well + ',' + transposonCoord + ',' + feature + ',' \
			+ averageValue + ',' + occupancy + ',' + occupancyTest + '\n'
			
			fileHandle.write(outputStr)
	
	fileHandle.close()
	
	return
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
def GenerateSudokuCatalogEntryWithOccupancyData(plateDictEntry, wellAverageValuesDict, \
wellOccupancyDict, wellOccupancyTestDict):
# Generates an updated sudoku catalog entry for a plate from image analysis of a storage plate image
	
	catalogEntryDict = {}

	for well in gWellList:
		
		catalogEntry = plateDictEntry[well]
		
		wellAverageValue = wellAverageValuesDict[well]
		wellOccupancy = wellOccupancyDict[well]
		wellOccupancyTest = wellOccupancyTestDict[well]
		
		catalogEntry.extend([wellAverageValue, wellOccupancy, wellOccupancyTest])
		
		catalogEntryDict[well] = catalogEntry
		
	return catalogEntryDict
# ------------------------------------------------------------------------------------------------ #
