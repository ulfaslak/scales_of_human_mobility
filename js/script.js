mapboxgl.accessToken = 'pk.eyJ1IjoiZGVlZ2dlIiwiYSI6ImNqM2Jmb29wYjAwN3kycXFrcW03YWlzdXAifQ.lRrvz11b3PTPopgJ888MMQ'; //public key

var map = new mapboxgl.Map({
	container: 'map',
	style: 'mapbox://styles/mapbox/basic-v9',
	zoom: 4,
	center: [12, 56]
});


//Get mapbox map canvas container
var canvas = map.getCanvasContainer();
var svg = d3.select(canvas).append("svg");

//Project any point to map's current state
function projectPoint(lon, lat) {
	return map.project(new mapboxgl.LngLat(lon, lat));
}

// Test point to render
var data = [[56, 12, 1, 0], [56, 11, 2, 10], [55, 10, 3, 2]]
 
// Color scale
var scaleColor = d3.scaleOrdinal()
	.domain([data.map(function(d) { return d[3]; })])
	.range(d3.range(data.length).map(function(d) {return d/data.length}))

// Circle template
var mapboxCircle = svg.selectAll("circle")
	.data(data).enter()
	.append("circle")
	.classed("stoploc", true)

// Update projection
function render() {
	mapboxCircle
		.attr('cx', function(d) { return projectPoint(d[1], d[0])['x']})
		.attr('cy', function(d) { return projectPoint(d[1], d[0])['y']})
		.attr('r', function(d) { return d[2]*10; })
		.attr('fill', function(d) { return d3.interpolateRainbow(scaleColor(d[3]))})
}

// Re-render visualization when view changes
map.on("move", function() {
  	render()
})

render();