# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-11-05 10:52:09
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-11-05 10:53:05
import xml.etree.ElementTree as ET
import xml
import subprocess

"""
Write XML file for GATV server
"""
def writeGatv(parking, results):

		id_respuesta = 1
		id_tipo = 1
		id_cam = 8
		limit = 10
		zone1_empty    = 0
		zone1_occupied = 0
		zone2_empty    = 0
		zone2_occupied = 0

		if 'C1' in parking:
			id_zone = 54
			zone1_occupied = sum(results[:limit])
			zone1_empty    = len(results[:limit]) - zone1_occupied
			zone2_occupied = sum(results[limit:])
			zone2_empty    = len(results[limit:]) - zone2_occupied

		elif 'C2' in parking:
			id_zone = 56
			zone1_occupied = sum(results[:limit-1])
			zone1_empty    = len(results[:limit-1]) - zone1_occupied
			zone2_occupied = sum(results[limit-1:])
			zone2_empty    = len(results[limit-1:]) - zone2_occupied

		
		root = ET.Element('Respuesta',{'id_respuesta':str(id_respuesta)})
		zone1 = ET.SubElement(root,'Zona',{'id_camara':str(id_cam),'id_zona':str(id_zone)})
		tipo1 = ET.SubElement(zone1,'Tipo',{'id_tipo':str(id_tipo)})
		ET.SubElement(tipo1,'Libres').text   = str(zone1_empty) 
		ET.SubElement(tipo1,'Ocupados').text = str(zone1_occupied)
		zone2 = ET.SubElement(root,'Zona',{'id_camara':str(id_cam),'id_zona':str(id_zone+1)})
		tipo2 = ET.SubElement(zone2,'Tipo',{'id_tipo':str(id_tipo)})
		ET.SubElement(tipo2,'Libres').text   = str(zone2_empty) 
		ET.SubElement(tipo2,'Ocupados').text = str(zone2_occupied)
		tree = ET.ElementTree(root)
		return tree

"""
Send XML to server
"""
def sendGatv(parking,results,api_server):

	tree = writeGatv(parking,results)

	xml_filename = 'config/plazas.xml'
	with open(xml_filename, "wb") as xml_file:
		tree.write(xml_file)
	try:
		php_command ="php php/placeupdate.php "+xml_filename+" php/NuevaApiKey php/resultado.xml "+api_server
		process = subprocess.check_call(php_command.split())
	except:
		print('Error during XML sending process')
