# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-10-05 12:34:51
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-10-05 12:53:51

"""
TEST MODE
"""

"""
Print accuracy for each parking lot
"""
def printTestResults(contours,results,img_csv_line,img_w,img_h):

	correctas = 0.00000
	fails = []
	for i in range(results):
		position = img_csv_line[results[i]]
		print('valor correcto: '+str(position))
		if(str(position) != str(results[i])):
			print('ERROR DE PREDICCION')
			fails.append(results[i])
		else:
			print('PREDICCION CORRECTA')
			correctas += 1.00000
		fails = np.array(fails)
	
	final = float((correctas/float(contours.shape[0]))*100)
	print('POSICIONES CORRECTAS = '+str(final)+' %')
	return final,fails

"""
If we have the original occupation labels, we can have the accuracy with a precise value,
not with human evaluation
"""
def testAccuracy(detector):

	img_line = detector.getLine(detector.args.image_path)
	ocup_filename = 'occupancies/occupancies_'+detector.parking+'.txt'
	ocup_file     = open(ocup_filename)
	lines_csv     = ocup_file.readlines()
	if(ocup_filename == 'occupancies/occupancies_C2_2734.txt' or
	   ocup_filename == 'occupancies/occupancies_U_3795.txt'):
		img_csv_line = lines_csv[detector.args.image_path].strip().split(',')
	else:
		img_csv_line = lines_csv[detector.args.image_path].strip().split('\t')
	
	final,fails = printTestResults(detector.contours,results,img_csv_line,img_w,img_h)

	saveResults(final,detector.args.image_path,img_w,img_h,fails)

"""
Stores results from prediction, such as predicted image, date, positions, % acc ...
"""
def saveTestResults(detector,final,img_analysed,img_w,img_h,fails):
	
	fecha = "{:%H_%M-%d_%m}".format(datetime.now())
	model_path = detector.getModelPath().split('/')
	model_name = model_path[len(model_path)-1]
	print(model_name)
	
	if not os.path.exists('models/all_parkings/'+model_name+'/predictions'):
		os.makedirs('models/all_parkings/'+model_name+'/predictions/text_data')
		os.makedirs('models/all_parkings/'+model_name+'/predictions/images')
		
	predict_file = 'models/all_parkings/'+model_name+'/predictions/text_data/'+fecha+".txt"
	cv2.imwrite('models/all_parkings/'+model_name+'/predictions/images/'+fecha+".png",detector.image)
		
	with open(predict_file, "a") as fileTxt:		
		fileTxt.write('IMAGEN ANALIZADA: '+img_analysed+"\n")
		fileTxt.write('POSICIONES CORRECTAS: '+str(final)+' % \n')
		fileTxt.write('MODELO USADO: '+str(model_name)+'\n')
		fileTxt.write('DIMENSIONES DE IMGs: '+str(img_w)+'x'+str(img_h)+'\n')
		fileTxt.write('ARCHIVO DE PUNTOS USADO: '+str(detector.puntos)+'\n')
		if(fails.size > 0):
			for fail in fails:
				fileTxt.write('PLAZA '+str(fail)+' INCORRECTA'+'\n')