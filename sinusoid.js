svgWidth=640;
svgHeight=450;
margin = {top:50, bottom:100, right:25, left:325}

numSliders = 4;
sliderRange = [[-3, 3], [-3, 3], [-2, 2], [-4, 4]];
sliderSpacing = (svgHeight - margin.bottom - margin.top)/4;
sliderxvals = [100, 100, 100, 100];
parameterVals = [1, 1, 0, 0];
slidermargin = {left:20, right:20}
axisLabels = ['A', 'B', 'C', 'D'];


svg = d3.select("#sinusoid_plot")
	.attr("width", svgWidth)
	.attr("height", svgHeight)

grp = svg.append("g")
	.attr("transform", "translate("+margin.left + ", " + margin.top + ")");

functionText = svg.append('g')
	.attr("transform", "translate("+(margin.left + (svgWidth - margin.left - margin.right)/2) + "," + (svgHeight - (margin.bottom/2)) + ")")
	.append("text").style('text-anchor', 'middle').text("f(x) = 1.0sin(1.0x + 0.0) + 0.0");

function f(x, a, b, c, d) {
	return (parameterVals[0] * Math.sin(parameterVals[1]*x + parameterVals[2]) + parameterVals[3]);
}

xMin = -10;
xMax = 10;



//----------------------------INITIALIZE DATA----------------------------------------
data = [];

for (var i = 0; i <= 100; i++){
	data.push({x: xMin + (xMax - xMin)*i*.01, y: f(xMin + (xMax - xMin)*i*.01, 1, 1, 0, 0) });
	}



//-----------------------------INITIALIZE SCALES---------------------------------------
x = d3.scaleLinear()
	.domain([xMin, xMax])
	.range([0, svgWidth - margin.left - margin.right]);

y = d3.scaleLinear()
	.domain([-4, 4])
	.range([svgHeight - margin.bottom - margin.top, 0])


var lineFunction = d3.line()
                         .x(function(d) { return x(d.x); })
                         .y(function(d) { return y(d.y); });

// data.forEach(function(d) {
// 	grp.append("circle")
// 	.attr("cx", x(d.x))
// 	.attr("cy", y(d.y))
// 	.attr("r", 4)
// 	.attr("fill", "red")
// })
var lineGraph = grp.append("path")
					.attr("d", lineFunction(data))
					.attr("stroke", "blue")
					.attr("stroke-width", 2)
					.attr("fill", "none");

//-------------------------------------DEFINE FUNCTIONS FOR UPDATING DATA------------------------------

plotxAxis = d3.axisBottom(x);
plotxAxisgrp = grp.insert("g", ":first-child")
	.attr("transform", "translate(0,"+((svgHeight - margin.top - margin.bottom)/2)+")")
	.call(plotxAxis)

plotyAxis = d3.axisRight(y);
plotyAxisgrp = grp.insert("g", ":first-child")
	.attr("transform", "translate("+((svgWidth - margin.left - margin.right)/2)+",0)")
	.call(plotyAxis)

//calculates the data for y = Asin(Bx+C) + D
function calculateData(a,b,c,d){
	for (var i = 0; i <= 100; i++){
		//data[i].y = f(xMin + (xMax - xMin)*i*.01, a, b, c, d);
		data[i].y = f(xMin + (xMax - xMin)*i*.01);
		}
}

function recomputeAxes(){
dataMin = d3.min(data, function(d) {return d.y});
dataMax = d3.max(data, function(d) {return d.y});

y = d3.scaleLinear()
	.domain([Math.min(dataMin, Math.max(-4, dataMax - 8)), Math.max(dataMax, Math.min(4, dataMin+8))])
	.range([svgHeight - margin.bottom - margin.top, 0])
plotyAxis = d3.axisRight(y);

plotxAxisgrp.attr("transform", "translate(0,"+(y(0))+")")
}

function updateFunctionText(){
	str = "f(x) = " + parameterVals[0].toFixed(1) + "sin("+parameterVals[1].toFixed(1)+"x";
	if (parameterVals[2].toFixed(1) < 0){ str = str + "-" + (-parameterVals[2].toFixed(1))} else {str = str + "+" + parameterVals[2].toFixed(1)}
	str += ") ";
	if (parameterVals[3].toFixed(1) < 0){ str = str + "-" + (-parameterVals[3].toFixed(1))} else {str = str + "+" + parameterVals[3].toFixed(1)}		
	functionText.text(str)
}

function drawData(){
	plotyAxisgrp.call(plotyAxis);
	lineGraph.attr("d", lineFunction(data))
	//grp.selectAll("circle").each(function(d, i){d3.select(this).attr("cy", y(data[i].y))});
}



var slidersquare = svg.append("g").attr("transform", "translate(0,"+ margin.top +")")

var sliderscales= [];
for (var i = 0; i < numSliders; i++){

	linegrp = slidersquare.append('g').attr("transform", "translate("+slidermargin.left+","+(40 + i*sliderSpacing)+")")
	linegrp.append("line")
			.attr("class", "track")
			.attr("x1", 0)
			.attr("x2", margin.left - slidermargin.right - slidermargin.left)
			.attr("y1", 0).attr("y2", 0)
			.call(d3.drag()
				.on("start",function(i){return function() {moveslider(i, d3.event.x); console.log(i)}}(i))
				.on("drag",function(i){return function() {moveslider(i, d3.event.x);}}(i))
				);
	linegrp.insert("line", ".track").attr("class", "track-inset")
			.attr("x1", 0)
			.attr("x2", margin.left - slidermargin.right - slidermargin.left)
			.attr("y1", 0).attr("y2", 0);	


	linegrp.insert("circle", ".track").attr("id", "handle" + i)
			.attr("class", "handles")
			.attr("r", 9)
			.attr("cx", d3.scaleLinear().domain(sliderRange[i]).range([0, margin.left-slidermargin.right-slidermargin.left])(parameterVals[i]))
			.attr("cy", 0);

	linegrp.append("text").attr('x', -15).attr('y', 10).style('text-anchor', 'middle').text(axisLabels[i]);

	sliderscales.push(d3.scaleLinear().domain([0, margin.left-slidermargin.right-slidermargin.left]).range(sliderRange[i]));

	xAxis = d3.axisBottom(d3.scaleLinear().domain(sliderRange[i]).range([0, margin.left-slidermargin.right-slidermargin.left]));

	slidersquare.insert('g', ":first-child")
		.attr('transform', 'translate('+slidermargin.left+','+(45+(i*sliderSpacing))+')')
		.attr('class', 'axis')
		.call(xAxis)
}



function moveslider(sliderNumber, xval){
	var truex = 0;
	h = d3.select("#handle" + sliderNumber);
	if (xval < margin.left-slidermargin.right){ h.attr("cx", xval); sliderxvals[sliderNumber] = xval;}
	if (xval < 0){h.attr("cx", 0); sliderxvals[sliderNumber] = 0;}
	if (xval > margin.left-slidermargin.right-slidermargin.left){
		h.attr("cx", margin.left-slidermargin.right-slidermargin.left);
		sliderxvals[sliderNumber] = margin.left - slidermargin.right-slidermargin.left;
		}
	parameterVals[sliderNumber] = sliderscales[sliderNumber](sliderxvals[sliderNumber])
	//calculateData(1, sliderscales[1](truex), 0, 0);
	calculateData();
	recomputeAxes();
	updateFunctionText();
	drawData();

}


