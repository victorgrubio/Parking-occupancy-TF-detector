<?php
//Carlos Alberto Martín Edo
//Grupo de Aplicación de Telecomunicaciones Visuales
//Universidad Politécnica de Madrid
//06 de noviembre de 2013

//Esta versión tiene funcionalidad completa y se parece mucho a la versión final. Recibe por línea de comandos los parámetros para la ejecución
//e intenta la comunicación por el puerto serie (plan B) si es que falla la comunicación directa por Internet (servicio 3.1.1.9). 
//La nueva funcionalidad de esta versión es que manda el camera_id, que coincide con el upv_id. 
//Es un valor que leemos del XML, por lo que ha habido que cambiar la especificación de este último.

//En esta versión, además, la URL raíz del servidor WePark se pasa por línea de comandos como un argumento más.

//En primer lugar, comprobamos el número de argumentos, que puede ser 3 o 4 (más el nombre del script PHP, esto es, en total 4 o 5).
if ((count($argv)<5) || (count($argv)> 6)){
    echo ("Número de argumentos no válido.\r\n");
    echo ("Es preciso pasar 4 o 5 argumentos:\r\n-el nombre del fichero XML que contiene la información de actualización\r\n");
    echo ("-el nombre del fichero con las credenciales de usuario\r\n-el nombre del fichero donde se guardará el resultado (Ok o error)\r\n");
    echo ("-la URL raíz del servidor WePark, terminada en / \r\n");
    echo ("-adicionalmente, se puede indicar el puerto para las comunicaciones serie, en su caso. Por ejemplo: /dev/ttyS0 o /dev/ttyUSB0\r\n");
    echo ("Saliendo...\r\n");
    exit(0);
}

//Abro el XML con los datos de las plazas. Los que me han pasado de prueba se llamaban respuesta-xxxx
if (file_exists($argv[1])){

//SimpleXML es una biblioteca para manipular ficheros XML
    $respuesta = simplexml_load_file($argv[1]);

    $error=0;
//Vamos iterando por cada bloque. La etiqueta es respuesta. Dentro del bloque "respuesta", tomamos la info de una zona.
//Va a haber (o puede haber) varias zonas por "respuesta" y cada zona tendrá varios tamaños.
//No puede haber varias respuestas, ya que equivaldría a tener info actualizada y antigua sobre unas mismas zonas.
    foreach ($respuesta->Zona as $zona){

//Nótese cómo algunas informaciones en el XML van como atributos dentro de la etiqueta. Es el caso del id de zona. Se recupera la info con la función attributes()
//También es el caso del camera_id, que coincide con el UPV_id. Sólo se emplea en el envío de los datos por el puerto serie.

	$zone_attibutes = $zona -> attributes();

	$zone_id = intval($zone_attibutes -> id_zona);
	$camera_id = intval($zone_attibutes -> id_camara);


//Traza para saber en qué zona estoy
//	print_r("\nInfo de la zona de id ".$zone_id."\n");

//Itero por los distintos tamaños de plazas. He mantenido la nomenclatura alusiva al tipo que aparece en nuestro XML (y no hablo de tamaño o size)
//por coherencia con el fichero XML.
	foreach ($zona->Tipo as $tipo){  
	    $size_id = intval($tipo -> attributes());
	    $plazaslibres = intval($tipo->Libres);
	    $plazasocupadas = intval($tipo->Ocupados);
//Trazas para verificar el funcionamiento del programita
//	    print_r("Tamaño de la zona: ".$size_id."\n");
//	    print_r("Plazas libres: ".$plazaslibres."\n");
//	    print_r("Plazas libres: ".$plazasocupadas."\n");

//Creo un array asociativo con la información, de acuerdo con la sintaxis del "servicio de actualización de plazas
//de aparcamiento (3.1.1.9).
	    $vector = array("zone"=>$zone_id, "size"=>$size_id, "free"=>$plazaslibres, "occupied"=>$plazasocupadas);
//Y a continuación lo codifico como json.	     
	    $json_plazas=json_encode($vector);
//Traza para verificar la pinta del json
//	    print_r($json_plazas."\r\n");

//Abro el fichero que ha de contener la ApiKey, que es el segundo argumento del script (el número [2], siendo el [0] el script)
	    if (!($credenciales=fopen($argv[2],'r'))) {
		echo ("No existe el fichero con la ApiKey.\r\n");
		echo ("El fichero ha de contener: usuario:ApiKey\r\n");
		echo ("Saliendo...\r\n");
		exit(0);
	    }

	    $apikey=fread($credenciales, filesize($argv[2]));
	    fclose($credenciales);


//La URL del servicio 3.1.1.9
	    $URL=$argv[4].'stats/zone/?format=json';

//Utilizo este "atajo" para la generación de parte de las cabeceras de la petición HTTP
	    ini_set('user_agent', "PHP\r\nAuthorization: ApiKey ".$apikey);

//Y de esta forma generamos el resto, con las opciones de contexto del transporte HTTP en PHP
	    $opts = array('http' =>
		array(
		    'method' => 'POST',
//Estas opciones comentadas pueden ser de interés en el futuro para indicar que el servidor se autoautentica
//	'verify_peer'=>true,
//	'allow_self_signed'=>true,
//	'protocol_version' => 1.1,
		    'header' => "Content-type: application/json\r\n",
		    'content' => $json_plazas
		)
	    );

//Creo el contexto a partir del array asociativo que acabo de generar
	    $context = stream_context_create($opts);

//Y abro la conexión como si fuese un fichero, con el contexto que acabo de crear. También se puede utilizar fopen(),
//que lleva un argumento maś.
//Esta llamada se hace tantas veces como pares zona-tamaño(tipo) haya. Observar los dos bucles "foreach" anidados.
	    $jsonresult = file_get_contents($URL, false, $context);

	    $result = json_decode($jsonresult);
//Traza para comprobar lo que nos ha devuelto el servicio.
//print_r($result);

	    $statusdevuelto = $result -> status;

//Esta condición se evalúa incluso aunque falle la invocación al servidor. En ese caso, $jsonresult no es un resultado, ni se puede descodificar en
//$result, ni puedo extraer el estatus. Ahora bien, la condición que vemos a continuación se evaluaría y diría que, en efecto, el status no es Ok.
//De esta forma, este fragmento de código nos sirve no sólo si no devuelve Ok, sino también si hay errores más graves, como http 401 o http 500.
	    if ($statusdevuelto != 'Ok') {
		$error++;
		if (!($puertoserie=fopen($argv[5],"r+"))) {
		    echo ("No existe o no se ha indicado el fichero especial que emula el puerto serie.\r\n");
		    echo ("Se tiene que parecer a /dev/ttyS0 o /dev/ttyUSB0 o /dev/usb/tty0 o similar\r\n");
		    echo ("Saliendo...\r\n");
		    $ficheroresultado = fopen($argv[3], 'w+');  
		    fwrite($ficheroresultado, '<?xml version="1.0" encoding="UTF-8"?>'."\r\n".'<!DOCTYPE WePark>'."\r\n".'<ResultadoActualizaPlazas>'."\r\n".'<status>Failure</status>'."\r\n".'</ResultadoActualizaPlazas>'."\r\n");
		    fclose($ficheroresultado);
		    exit(0);
		}

//Si llego a este punto, es que el nombre que me han pasado para el dispositivo existe, aunque podría ser cualquier cosa.
//Preparo el dispositivo serie con /bin/stty
//La opción -F precede al dispositivo; 57600 es la velocidad en baudios - hay un conjunto limitado de velocidades
//configurables; -cstopb hace que haya un bit de stop (sin el signo -, habría 2); cs8 significa caracteres de 8 bits;
//-parenb es negar que se genere un bit de paridad; raw permite el modo de entrada en bruto; cread y clocal las he puesto
//porque vienen en el ejemplo del manual, pero no estoy de que hagan algo que nos sirva
		exec("/bin/stty -F ".$argv[5]." 57600 -cstopb cs8 -parenb raw cread clocal");

//serie es un sencillo programa en C que utiliza las funcionalidades provistas por la biblioteca wpTLV.
//En concreto crea un fichero con la codificación UART de lo que pasemos
//Recordar que el camera_id coincide con el UPV_id y que lo hemos extraído del XML de plazas
		exec("./serie ".$camera_id." ".$zone_id." ".$size_id." ".$plazaslibres." ".$plazasocupadas." cadenauart");

//cadenauart es el fichero que se crea en el exec anterior
		$uart = "cadenauart";
		$ficherouart = fopen($uart, "r");
		$contenidouart = fread($ficherouart, filesize($uart));

		fclose($ficherouart);


//¡¡Muy importante!! ¡Los permisos para escribir en el puerto serie! Típicamente, bastará con que el usuario esté en el grupo dialout.
		fwrite($puertoserie, $contenidouart);

		fclose ($puertoserie);

	    }

	}

    }

//Si la variable de control $error es mayor que 0, significa que no ha tenido el Ok del servidor para al menos un par zona-tamaño. En ese caso, marco el status como serie.
//Si no, es que es 0, luego todo ha ido bien.
    if ($error > 0) {
	$ficheroresultado = fopen($argv[3], 'w+');  
	fwrite($ficheroresultado, '<?xml version="1.0" encoding="UTF-8"?>'."\r\n".'<!DOCTYPE WePark>'."\r\n".'<ResultadoActualizaPlazas>'."\r\n".'<status>serie</status>'."\r\n".'</ResultadoActualizaPlazas>'."\r\n");
	fclose($ficheroresultado);
    } else {
	$ficheroresultado = fopen($argv[3], 'w+');  
	fwrite($ficheroresultado, '<?xml version="1.0" encoding="UTF-8"?>'."\r\n".'<!DOCTYPE WePark>'."\r\n".'<ResultadoActualizaPlazas>'."\r\n".'<status>Ok</status>'."\r\n".'</ResultadoActualizaPlazas>'."\r\n");
	fclose($ficheroresultado);
    }

} else 
    echo ("El fichero de plazas libres y ocupadas no existe\r\n");
    exit (0);

?>
