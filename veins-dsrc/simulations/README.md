1.
Export OpenStreetMaps

2. 
```
netconvert --osm-files highway.osm --output-file highway.net.xml --geometry.remove --roundabouts.guess --ramps.guess --junctions.join --tls.guess-signals --tls.discard-simple --tls.join
```

3.
```
python randomTrips.py -n ../../../../Users/btang/Documents/sumo/downtown.net.xml -e 10000 -o ../../../../Users/btang/Documents/sumo/downtown.trips.xml
```

4.
```
duarouter -n downtown.net.xml --route-files downtown.trips.xml -o downtown.rou.xml --ignore-errors
```

5. `downtown.sumo.cfg` contains
```
	<?xml version="1.0" encoding="iso-8859-1"?>
	
	<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" csi:noNamespaceSchemaLocation="http://sumo.sf.net/xsd/sumoConfiguration.xsd">
	
		<input>
			<net-file value="downtown.net.xml"/>
			<route-files value="downtown.rou.xml"/>
		</input>
		
		<time>
			<begin value="0"/>
			<end value="1000"/>
		</time>
	</configuration>
```
