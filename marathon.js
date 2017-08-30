//-------------------------------------------PAGE SETUP---------------------------------------------------

var m1Parameters = {
svgWidth: 850,
svgHeight: 550,
margin: {top:35, bottom:125, right:50, left:50},
sliderYearMargin: {left:300, right:20},
sliderAgeMargin: {left:300, right:20},
plotWidth: 0,//m1Parameters.svgWidth - m1Parameters.margin.right - m1Parameters.margin.left,
plotHeight: 0,//m1Parameters.svgHeight - m1Parameters.margin.top - m1Parameters.margin.bottom,
sliderYearWidth: 0,//m1Parameters.svgWidth - m1Parameters.margin.left-m1Parameters.margin.right-m1Parameters.sliderYearMargin.left - m1Parameters.sliderYearMargin.right,
sliderAgeWidth: 0,//m1Parameters.svgWidth - m1Parameters.margin.left-m1Parameters.margin.right-m1Parameters.sliderAgeMargin.left - m1Parameters.sliderAgeMargin.right,
longTransitionTime:  800,
transitionTime: 0,
transitionRatio: 0.8,
mainColor: "lightseagreen",
years: [02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16],
ages: [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
yValues: [100, 300, 600, 1000, 2000, 3000],
buttonStatus: {femalemale: true, male:false, female:false, ed:true},
majorTimeTicks: [120, 180, 240, 300, 360],
minorTimeTicks: [135, 150, 165, 195, 210, 225, 255, 270, 285, 315, 330, 345, 375, 390],
backgroundButtonHeight: 28,
backgroundButtonWidth: 155
}

m1Parameters.plotWidth = m1Parameters.svgWidth - m1Parameters.margin.right - m1Parameters.margin.left;
m1Parameters.plotHeight = m1Parameters.svgHeight - m1Parameters.margin.top - m1Parameters.margin.bottom;
m1Parameters.sliderYearWidth = m1Parameters.svgWidth - m1Parameters.margin.left-m1Parameters.margin.right-m1Parameters.sliderYearMargin.left - m1Parameters.sliderYearMargin.right;
m1Parameters.sliderAgeWidth = m1Parameters.svgWidth - m1Parameters.margin.left-m1Parameters.margin.right-m1Parameters.sliderAgeMargin.left - m1Parameters.sliderAgeMargin.right;



//------------------------------------------PAREMETER SETUP----------------------------------------------

// const edYears = [2003, 2004, 2005, 2006, 2011, 2012, 2013, 2016];
// const edTimes = {2003: 179, 2004: 175, 2005: 183, 2006: 189, 2011: 196, 2012: 210, 2013: 222, 2016: 237};
// const edText = {2003: "2:59:08", 2004: "2:54:44", 2005: "3:02:37", 2006: "3:08:34",
// 				2011: "3:15:50", 2012: "3:30:25", 2013: "3:41:58", 2016: "3:56:34"};
//const ages = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]; //15 represents <20, 70 represents >65


var m1Variables = {
gridAbovePlot: 10,
bgButtonMargin: {left: 25},
numTicks: 60,

median: 0,

sexButtonWidth: 45,
sexButtonHeight: 25,

holdHistogram: false,
backgroundMax: 0,

globalData: null,
maskedGlobalData: null,
currentBins: null,
backgroundBins: null,


allYears: true,
year: 2016,
ageMin: m1Parameters.ages[0],
ageMax: m1Parameters.ages[m1Parameters.ages.length - 1],
yMax: 3000
}

//----------------------------------------APPEND PLOT GROUPS AND CONTROL GROUPS-------------------------
var m1svg = d3.select("#marathon_plot")
	.attr("width", m1Parameters.svgWidth)
	.attr("height", m1Parameters.svgHeight)
	.attr("background", "blue")

var m1plotGrp = m1svg.append("g").attr("transform", "translate("+m1Parameters.margin.left+","+m1Parameters.margin.top+")");
var m1ctrlGrp = m1svg.append('g').attr("transform", "translate("+m1Parameters.margin.left+","+(m1Parameters.svgHeight - m1Parameters.margin.bottom + 50)+")");

m1plotGrp.append("line")
	.attr("class", "medianLine").attr("y1", m1Parameters.plotHeight).attr("y2", -(m1Variables.gridAbovePlot)).attr("stroke-dasharray", "10,2");
// m1plotGrp.append("text").style('text-anchor', 'middle')
// 	.attr("class", "medianText").attr("y", -(m1Variables.gridAbovePlot + 20)).text("Median");


var m1SliderGroups = {
sliderYearGrp: m1ctrlGrp.append('g').attr("transform", "translate("+m1Parameters.sliderYearMargin.left+",0)"),
sliderAgeGrp: m1ctrlGrp.append('g').attr("transform", "translate("+m1Parameters.sliderAgeMargin.left+",50)")
}



var m1ButtonGroups = {
backgroundButton: m1ctrlGrp.append('g').attr("transform", "translate("+m1Variables.bgButtonMargin.left+","+(-m1Parameters.backgroundButtonHeight/2)+")"),
fmButton: m1ctrlGrp.append('g').attr("transform", "translate("+m1Variables.bgButtonMargin.left+","+(50-(m1Variables.sexButtonHeight/2)) +")"),
fButton: m1ctrlGrp.append('g').attr("transform", "translate("+(m1Variables.bgButtonMargin.left+(m1Variables.sexButtonWidth + 10))+","+(50-(m1Variables.sexButtonHeight/2)) +")"),
mButton: m1ctrlGrp.append('g').attr("transform", "translate("+(m1Variables.bgButtonMargin.left+(2*(m1Variables.sexButtonWidth + 10))) +","+(50-(m1Variables.sexButtonHeight/2)) +")"),
//eButton: m1ctrlGrp.append('g').attr("transform", "translate("+(m1Variables.bgButtonMargin.left+(3*(m1Variables.sexButtonWidth + 10))) + ","+(50-(m1Variables.sexButtonHeight/2)) +")")
}

//-----------------------------------------------SLIDERS----------------------------------------------------------------

var m1Scales = {
x: d3.scaleLinear().domain([120, 390]).range([0, m1Parameters.plotWidth]),
y: d3.scaleLinear().domain([0, m1Variables.yMax]).range([m1Parameters.plotHeight, 0]),

xYearSlider: d3.scaleLinear().domain([m1Parameters.years[0] - 2, m1Parameters.years[m1Parameters.years.length - 1]]).range([0, m1Parameters.sliderYearWidth]),
xYearSliderInv: d3.scaleLinear().range([m1Parameters.years[0] - 2, m1Parameters.years[m1Parameters.years.length - 1]]).domain([0, m1Parameters.sliderYearWidth]),

xAgeSlider: d3.scaleLinear().domain([m1Parameters.ages[0], m1Parameters.ages[m1Parameters.ages.length - 1]]).range([0, m1Parameters.sliderAgeWidth]),
xAgeSliderInv: d3.scaleLinear().range([m1Parameters.ages[0], m1Parameters.ages[m1Parameters.ages.length - 1]]).domain([0, m1Parameters.sliderAgeWidth]),

yAxis: null
}
m1Scales.yAxis = d3.axisLeft(m1Scales.y).tickSize(m1Parameters.plotWidth)

d3.csv("http://gsscott.github.io/df1veryslim2002.csv", function(data) {
	m1Variables.globalData = data;
	m1Variables.maskedGlobalData = data;
	console.log("Done loading");
	m1initializeXTicks();
	m1plotGrp.append('g').attr("class", "yaxis")
		.attr("transform", "translate("+(m1Parameters.plotWidth)+",0)").call(m1Scales.yAxis.ticks(3));
	m1plotGrp.select(".yaxis").selectAll(".tick line").attr("stroke", "#777")
		.attr("stroke-dasharray", "2,2");
	m1plotGrp.select(".yaxis").select(".domain").remove();
	m1plotHistogram();
	m1Parameters.transitionTime = m1Parameters.longTransitionTime;
	m1initializeControls();
});





function m1maskglobalData(){
	if (m1Variables.allYears){
		m1Variables.maskedGlobalData = m1Variables.globalData;
	}
	else{
		m1Variables.maskedGlobalData = m1Variables.globalData.filter(function(d){ return d['Year'] == m1Variables.year;})
	}
	m1Variables.maskedGlobalData = m1Variables.maskedGlobalData.filter(function(d){ return d['Min age'] >= m1Variables.ageMin;})
	m1Variables.maskedGlobalData = m1Variables.maskedGlobalData.filter(function(d){ return d['Max age'] <= m1Variables.ageMax;})
	if (m1Parameters.buttonStatus.female){
		m1Variables.maskedGlobalData = m1Variables.maskedGlobalData.filter(function(d){ return (d['Sex'] == "F");})
	}
	if (m1Parameters.buttonStatus.male){
		m1Variables.maskedGlobalData = m1Variables.maskedGlobalData.filter(function(d){ return (d['Sex'] == "M");})
	}	
}

function m1setBackground(){
	m1Variables.holdHistogram = !m1Variables.holdHistogram;
	if (m1Variables.holdHistogram==1){
		m1ButtonGroups.backgroundButton.select("rect").attr("fill", "black");
		m1ButtonGroups.backgroundButton.select("text").attr("fill", "#eeeeee");
		m1Variables.backgroundBins = d3.histogram().domain(m1Scales.x.domain()).thresholds(m1Scales.x.ticks(m1Variables.numTicks))(m1Variables.maskedGlobalData.map(function (d){return parseInt(d['Chip Time (min)'])}));
		m1Variables.backgroundMax = d3.max(m1Variables.backgroundBins, function(d){return d.length});
		m1plotGrp.selectAll(".bar")
			.transition().duration(m1Parameters.transitionTime)
				.attr("transform", function(d) {return "translate("+Math.ceil(m1Scales.x(d.x0))+","+m1Parameters.plotHeight+")";})
				.attr("height", 0);	
		m1plotBackgroundHistogram();
		}
	else {
		m1ButtonGroups.backgroundButton.select("rect").attr("fill", m1Parameters.mainColor);
		m1ButtonGroups.backgroundButton.select("text").attr("fill", "black");
		m1Variables.backgroundMax=0;
		
		//m1plotGrp.selectAll(".foregroundBar").remove();
		var oldyMax = m1Variables.yMax;
		m1Variables.currentBins = d3.histogram().domain(m1Scales.x.domain()).thresholds(m1Scales.x.ticks(m1Variables.numTicks))(m1Variables.maskedGlobalData.map(function (d){return parseInt(d['Chip Time (min)'])}));
		m1Scales.y.domain([0, m1calculateNewyMax(Math.max(m1Variables.backgroundMax, d3.max(m1Variables.currentBins, function(d){return d.length;})))]);
		if (oldyMax == m1Variables.yMax){m1plotGrp.selectAll(".backgroundBar").remove();}
		if (oldyMax > m1Variables.yMax){
			m1plotGrp.selectAll(".backgroundBar")
				.transition().duration(m1Parameters.transitionTime)
					.attr("transform", function(d) {return "translate("+Math.ceil(m1Scales.x(d.x0))+","+m1Parameters.plotHeight+")";})
					.attr("height", 0);
			d3.transition().duration(m1Parameters.transitionTime).ease(d3.easeSin).on("end", function() {	
			var merged = m1plotGrp.selectAll(".bar")
				 	m1plotGrp.select(".yaxis").transition().duration((m1Parameters.transitionTime*m1Parameters.transitionRatio)).ease(d3.easeSin).call(m1Scales.yAxis);
				 	m1plotGrp.select(".yaxis").selectAll(".tick line").attr("stroke", "#777")
				 		.attr("stroke-dasharray", "2,2");
				 	m1plotGrp.select(".yaxis").select(".domain").remove();	
					merged.transition().duration((m1Parameters.transitionTime*m1Parameters.transitionRatio)).ease(d3.easeSin)
							.attr("transform", function(d) {return "translate("+Math.ceil(m1Scales.x(d.x0))+","+m1Scales.y(d.length)+")";})
							.attr("width", function(d) {return Math.ceil(m1Scales.x(d.x1) - m1Scales.x(d.x0) +0);})
							.attr("height", function(d) {return ( m1Parameters.plotHeight - m1Scales.y(d.length));})	})		

		 }
		//m1plotHistogram();		
	}
}


function m1plotBackgroundHistogram(){
	console.log("m1plotBackgroundHistogram called");
	var bins = d3.histogram().domain(m1Scales.x.domain()).thresholds(m1Scales.x.ticks(m1Variables.numTicks))(m1Variables.maskedGlobalData.map(function (d){return parseInt(d['Chip Time (min)'])}))
	m1Variables.backgroundMax = d3.max(bins, function(d){return d.length});
	var bar = m1plotGrp.selectAll(".backgroundBar").data(bins);	
	bar.enter().insert("rect", ".medianLine")
			.attr("class", "backgroundBar")
			.attr("fill","#f5b4af")
			.attr("opacity", 0.5)
		.merge(bar)
			.attr("transform", function(d) {return "translate("+Math.ceil(m1Scales.x(d.x0))+","+m1Scales.y(d.length)+")";})
			.attr("width", function(d) {return Math.ceil(m1Scales.x(d.x1) - m1Scales.x(d.x0) +0);})
			.attr("height", function(d) {return ( m1Parameters.plotHeight - m1Scales.y(d.length));});		
}

function m1plotHistogram(){
	var oldyMax = m1Variables.yMax;
	var oldBins = m1Variables.currentBins;
	m1Variables.currentBins = d3.histogram().domain(m1Scales.x.domain()).thresholds(m1Scales.x.ticks(m1Variables.numTicks))(m1Variables.maskedGlobalData.map(function (d){return parseInt(d['Chip Time (min)'])}));
	m1Variables.median = d3.median(m1Variables.maskedGlobalData.map(function(d){return parseInt(d['Chip Time (min)'])}))
	m1Scales.y.domain([0, m1calculateNewyMax(Math.max(m1Variables.backgroundMax, d3.max(m1Variables.currentBins, function(d){return d.length;})))]);


	if (oldyMax < m1Variables.yMax){ 
		console.log("expanding");
		//sliderYearGrp.select(".trackborder").attr("pointer-events", "none")
		var yTemp = d3.scaleLinear().domain([0, oldyMax]).range([m1Parameters.plotHeight, 0])
		var bar = m1plotGrp.selectAll(".bar").data(oldBins);	
		var merged = bar.enter().insert("rect", ".medianLine")
				.attr("class", "bar")
				.attr("x", 0)
				.attr("fill",m1Parameters.mainColor)
				.attr("opacity", 0.9)
			.merge(bar);	
		if (m1Variables.holdHistogram){
				m1plotGrp.selectAll(".backgroundBar")
					.transition().duration((m1Parameters.transitionTime*m1Parameters.transitionRatio))
					.attr("transform", function(d) {return "translate("+Math.ceil(m1Scales.x(d.x0))+","+m1Scales.y(d.length)+")";})
					.attr("width", function(d) {return Math.ceil(m1Scales.x(d.x1) - m1Scales.x(d.x0) +0);})
					.attr("height", function(d) {return ( m1Parameters.plotHeight - m1Scales.y(d.length));});
				}		
		merged.transition().duration((m1Parameters.transitionTime*m1Parameters.transitionRatio))
			.attr("transform", function(d) {return "translate("+Math.ceil(m1Scales.x(d.x0))+","+m1Scales.y(d.length)+")";})
			.attr("width", function(d) {return Math.ceil(m1Scales.x(d.x1) - m1Scales.x(d.x0) +0);})
			.attr("height", function(d) {return ( m1Parameters.plotHeight - m1Scales.y(d.length));});			
		m1plotGrp.select(".yaxis").transition().duration((m1Parameters.transitionTime*m1Parameters.transitionRatio)).ease(d3.easeSin).call(m1Scales.yAxis).on("end", function() { console.log("redrawing");
		bar = m1plotGrp.selectAll(".bar").data(m1Variables.currentBins);	
		merged = bar.enter().insert("rect", ".medianLine")
				.attr("class", "bar")
				.attr("x", 0)
				.attr("fill",m1Parameters.mainColor)
				.attr("opacity", 0.9)
			.merge(bar);			
		merged.transition().duration((m1Parameters.transitionTime*m1Parameters.transitionRatio))
				.attr("transform", function(d) {return "translate("+Math.ceil(m1Scales.x(d.x0))+","+m1Scales.y(d.length)+")";})
				.attr("width", function(d) {return Math.ceil(m1Scales.x(d.x1) - m1Scales.x(d.x0) +0);})
				.attr("height", function(d) {return ( m1Parameters.plotHeight - m1Scales.y(d.length));});
		d3.select(".medianLine").transition().duration(m1Parameters.transitionTime*m1Parameters.transitionRatio).attr("x1", m1Scales.x(m1Variables.median)).attr("x2", m1Scales.x(m1Variables.median));
		})	
		m1plotGrp.select(".yaxis").selectAll(".tick line").attr("stroke", "#777")
			.attr("stroke-dasharray", "2,2");
		m1plotGrp.select(".yaxis").select(".domain").remove();
	}

	if (oldyMax == m1Variables.yMax){
		var bar = m1plotGrp.selectAll(".bar").data(m1Variables.currentBins);	
		var merged = bar.enter().insert("rect", ".medianLine")
				.attr("class", "bar")
				.attr("x", 0)
				.attr("fill",m1Parameters.mainColor)
				.attr("opacity", 0.9)
			.merge(bar);		
		merged.transition().duration(m1Parameters.transitionTime)
				.attr("transform", function(d) {return "translate("+Math.ceil(m1Scales.x(d.x0))+","+m1Scales.y(d.length)+")";})
				.attr("width", function(d) {return Math.ceil(m1Scales.x(d.x1) - m1Scales.x(d.x0) +0);})
				.attr("height", function(d) {return ( m1Parameters.plotHeight - m1Scales.y(d.length));});
		d3.select(".medianLine").transition().duration(m1Parameters.transitionTime).attr("x1", m1Scales.x(m1Variables.median)).attr("x2", m1Scales.x(m1Variables.median));
		//d3.select(".medianText").transition().duration(transitionTime).attr("x", x(m1Variables.median));
	}

	if (oldyMax > m1Variables.yMax){
		console.log("contracting");
		var yTemp = d3.scaleLinear().domain([0, oldyMax]).range([m1Parameters.plotHeight, 0])
		var bar = m1plotGrp.selectAll(".bar").data(m1Variables.currentBins);	
		var merged = bar.enter().insert("rect", ".medianLine")
				.attr("class", "bar")
				.attr("x", 0)
				.attr("fill",m1Parameters.mainColor)
				.attr("opacity", 0.9)
			.merge(bar);		
		merged.transition().duration((m1Parameters.transitionTime*m1Parameters.transitionRatio))
				.attr("transform", function(d) {return "translate("+Math.ceil(m1Scales.x(d.x0))+","+yTemp(d.length)+")";})
				.attr("width", function(d) {return Math.ceil(m1Scales.x(d.x1) - m1Scales.x(d.x0) +0);})
				.attr("height", function(d) {return ( m1Parameters.plotHeight - yTemp(d.length));});
		d3.select(".medianLine").transition().duration(m1Parameters.transitionTime*m1Parameters.transitionRatio).attr("x1", m1Scales.x(m1Variables.median)).attr("x2", m1Scales.x(m1Variables.median));
		d3.transition().duration((m1Parameters.transitionTime*m1Parameters.transitionRatio)).ease(d3.easeSin).on("end", function() {
				 	m1plotGrp.select(".yaxis").transition().duration((m1Parameters.transitionTime*m1Parameters.transitionRatio)).ease(d3.easeSin).call(m1Scales.yAxis);
				 	m1plotGrp.select(".yaxis").selectAll(".tick line").attr("stroke", "#777")
				 		.attr("stroke-dasharray", "2,2");
				 	m1plotGrp.select(".yaxis").select(".domain").remove();	
					merged.transition().duration((m1Parameters.transitionTime*m1Parameters.transitionRatio)).ease(d3.easeSin)
							.attr("transform", function(d) {return "translate("+Math.ceil(m1Scales.x(d.x0))+","+m1Scales.y(d.length)+")";})
							.attr("width", function(d) {return Math.ceil(m1Scales.x(d.x1) - m1Scales.x(d.x0) +0);})
							.attr("height", function(d) {return ( m1Parameters.plotHeight - m1Scales.y(d.length));})
					if (m1Variables.holdHistogram){
							m1plotGrp.selectAll(".backgroundBar")
								.transition().duration((m1Parameters.transitionTime*m1Parameters.transitionRatio))
								.attr("transform", function(d) {return "translate("+Math.ceil(m1Scales.x(d.x0))+","+m1Scales.y(d.length)+")";})
								.attr("width", function(d) {return Math.ceil(m1Scales.x(d.x1) - m1Scales.x(d.x0) +0);})
								.attr("height", function(d) {return ( m1Parameters.plotHeight - m1Scales.y(d.length));});
							}
					});		
	 	}
	}


function m1calculateNewyMax(y){
	console.log("in: "+y);
	for (var i = 0; i < m1Parameters.yValues.length; i++){
		if (y < m1Parameters.yValues[i]){
			y = m1Parameters.yValues[i];
			m1Variables.yMax = y;
			return y;
		}
	}
	return m1Parameters.yValues[m1Parameters.yValues.length - 1];
}

function m1initializeXTicks(){

	//----------------------------------------------DRAW THE TICKMARKS-----------------------------------
	m1plotGrp.insert("line", ".medianLine").attr("x1", -1).attr("x2", m1Parameters.plotWidth+1)
		.attr("y1", m1Parameters.plotHeight+1).attr("y2", m1Parameters.plotHeight+1).attr("class", "bottomLine");
	for (i = 0; i < (m1Parameters.majorTimeTicks.length); i++){
		console.log("adding line");
		m1plotGrp.insert("line", ".medianLine")
			.attr("x1", m1Scales.x(m1Parameters.majorTimeTicks[i])).attr("x2", m1Scales.x(m1Parameters.majorTimeTicks[i]))
			.attr("y1", -(m1Variables.gridAbovePlot)).attr("y2", m1Parameters.plotHeight)
			.attr("class", "majorTimeTick");
		m1plotGrp.append("text")
			.attr("x", m1Scales.x(m1Parameters.majorTimeTicks[i]))
			.attr("y", m1Parameters.plotHeight + 15)
			.attr("class", "majorTimeTickLabel")
			.style('text-anchor', 'middle')
			.text(parseInt(m1Parameters.majorTimeTicks[i]/60).toString() + ":00" );
	}
	for (i = 0; i < (m1Parameters.minorTimeTicks.length); i++){
		console.log("adding line");
		m1plotGrp.insert("line", ".medianLine")
			.attr("x1", m1Scales.x(m1Parameters.minorTimeTicks[i])).attr("x2", m1Scales.x(m1Parameters.minorTimeTicks[i]))
			.attr("y1", -(m1Variables.gridAbovePlot)).attr("y2", m1Parameters.plotHeight)
			.attr("class", "minorTimeTick");
	}

}



//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//---------------------------------------------DRAW CONTROL PANEL-----------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------






function m1initializeControls(){
	//----------------------------------------------DRAW THE 'YEAR' SLIDER-------------------------------
	m1SliderGroups.sliderYearGrp.append("line")
			.attr("class", "trackborder")
			.attr("x1", 0)
			.attr("x2", 0 + m1Parameters.sliderYearWidth)
			.attr("y1", 0).attr("y2", 0)
			.call(d3.drag()
				.on("start",function() {m1moveYearSlider(d3.event.x)})
				.on("drag",function() {m1moveYearSlider(d3.event.x)})
				);			
	m1SliderGroups.sliderYearGrp.insert("line", ".trackborder")
			.attr("class", "track")
			.attr("x1", 0)
			.attr("x2", 0 + m1Parameters.sliderYearWidth)
			.attr("y1", 0).attr("y2", 0);
	for (var i = 0; i < m1Parameters.years.length; i++){
		m1SliderGroups.sliderYearGrp.insert("circle", ".trackborder")
			.attr("class", "yearCircle")
			.attr("cx", ((i + 2) * (m1Parameters.sliderYearWidth/(m1Parameters.years.length+1))))
			.attr("r", 6)
		m1SliderGroups.sliderYearGrp.insert("text")
			.attr("x", ((i + 2) * (m1Parameters.sliderYearWidth/(m1Parameters.years.length+1))))
			.attr("y", 20).style('text-anchor', 'middle')
			.style("font-size", "11px")
			.text('20'+('00' + m1Parameters.years[i].toString()).slice(-2));
		}
	m1SliderGroups.sliderYearGrp.insert("circle", ".trackborder")
		.attr("class", "yearCircle")
		.attr("cx", 0)
		.attr("r", 6)
	m1SliderGroups.sliderYearGrp.insert("text")
		.attr("x", 0).attr("y", 20).style('text-anchor', 'middle').style("font-size", "11px")
		.text("All Years");
	m1SliderGroups.sliderYearGrp.insert("circle", ".trackborder")
			.attr("id", "m1yearHandle")
			.attr("r", 8)
			.attr("cx", 0)
			.attr("cy", 0);		
	m1SliderGroups.sliderYearGrp.append("text")
			.attr("text-anchor", "end")
			.attr("y", 5).attr("x", -15)
			.text("Year:")	

	//----------------------------------------DRAW THE 'AGE' SLIDER---------------------------------------
	m1SliderGroups.sliderAgeGrp.append("line")
			.attr("class", "trackborder")
			.attr("x1", 0)
			.attr("x2", 0 + m1Parameters.sliderAgeWidth)
			.attr("y1", 0).attr("y2", 0);
	m1SliderGroups.sliderAgeGrp.insert("line", ".trackborder")
			.attr("class", "track")
			.attr("x1", 0)
			.attr("x2", 0 + m1Parameters.sliderAgeWidth)
			.attr("y1", 0).attr("y2", 0);	
	m1SliderGroups.sliderAgeGrp.insert("line", ".trackborder")
			.attr("class", "m1selectedAgeTrack")
			.attr("x1", 0)
			.attr("x2", 0 + m1Parameters.sliderAgeWidth)
			.attr("y1", 0).attr("y2", 0);		
	for (var i = 0; i < m1Parameters.ages.length; i++){
		m1SliderGroups.sliderAgeGrp.insert("circle", ".trackborder")
			.attr("class", "yearCircle")
			.attr("cx", m1Scales.xAgeSlider(m1Parameters.ages[i]))
			.attr("r", 6)
		m1SliderGroups.sliderAgeGrp.insert("text")
			.attr("x", m1Scales.xAgeSlider(m1Parameters.ages[i]))
			.attr("y", 20).style('text-anchor', 'middle')
			.style("font-size", "11px")
			.text(m1Parameters.ages[i].toString());
		}
	m1SliderGroups.sliderAgeGrp.select("text").text("<20");	
	m1SliderGroups.sliderAgeGrp.select("text:last-of-type").text(">70");
	m1SliderGroups.sliderAgeGrp.append("polyline")
			.attr("class", "triangles")
			.attr("id", "m1ageLeftHandle")
			.attr("transform", "translate("+m1Scales.xAgeSlider(m1Variables.ageMin)+",0)")
			.attr("points", "-7,-12,12,0,-7,12")
			.call(d3.drag()
				.on("start drag", function() {m1moveLeftAgeSlider(d3.event.x)}));
	m1SliderGroups.sliderAgeGrp.append("polyline")
			.attr("class", "triangles")
			.attr("id", "m1ageRightHandle")
			.attr("transform", "translate("+m1Scales.xAgeSlider(m1Variables.ageMax)+",0)")
			.attr("points", "7,-12,7,12,-12,0")
			.call(d3.drag()
				.on("start drag", function() {m1moveRightAgeSlider(d3.event.x)}));
	m1SliderGroups.sliderAgeGrp.append("text")
			.attr("text-anchor", "end")
			.attr("y", 5).attr("x", -15)
			.text("Age:");

	//----------------------------------------------------DRAW BUTTONS--------------------------------------------------
	 m1ButtonGroups.backgroundButton.append("rect")
	 	.attr("height", m1Parameters.backgroundButtonHeight)
	 	.attr("width", m1Parameters.backgroundButtonWidth)
	 	.attr("rx", 5).attr("ry", 5)
	 	.attr("fill", m1Parameters.mainColor);

    m1ButtonGroups.backgroundButton.append("text")
		.attr("x", m1Parameters.backgroundButtonWidth/2)
		.attr("y", m1Parameters.backgroundButtonHeight/2 + 5)
		.attr("id", "m1buttonText")
		.style('text-anchor', 'middle').style('color', 'white')
		.text("Hold Current Plot");

	 m1ButtonGroups.backgroundButton.append("rect")
	 	.attr("height", m1Parameters.backgroundButtonHeight)
	 	.attr("width", m1Parameters.backgroundButtonWidth)
	 	.attr("rx", 5).attr("ry", 5)
	 	.style("opacity", 0)
     		.on('mousedown', function() { m1setBackground();});


	 m1ButtonGroups.fmButton.append("rect")
	 	.attr("height", m1Variables.sexButtonHeight).attr("width", m1Variables.sexButtonWidth)
	 	.attr("rx", 5).attr("ry", 5).attr("fill", "black")

     m1ButtonGroups.fmButton.append("text")
		.attr("x", m1Variables.sexButtonWidth/2).attr("y", m1Variables.sexButtonHeight/2 + 5)
		.attr("class", "sexButtonText").style('text-anchor', 'middle').attr("fill", "#eeeeee")
		.text('\uf182' + ' ' + '\uf183');

	 m1ButtonGroups.fmButton.append("rect")
	 	.attr("height", m1Variables.sexButtonHeight).attr("width", m1Variables.sexButtonWidth)
	 	.attr("rx", 5).attr("ry", 5).style("opacity", 0)
     		.on('mousedown', m1clickFMButton);

	 m1ButtonGroups.fButton.append("rect")
	 	.attr("height", m1Variables.sexButtonHeight).attr("width", m1Variables.sexButtonWidth)
	 	.attr("rx", 5).attr("ry", 5).attr("fill", m1Parameters.mainColor)

     m1ButtonGroups.fButton.append("text")
		.attr("x", m1Variables.sexButtonWidth/2).attr("y", m1Variables.sexButtonHeight/2 + 5)
		.attr("class", "sexButtonText").style('text-anchor', 'middle').attr("fill", 'black')
		.text('\uf182');

	 m1ButtonGroups.fButton.append("rect")
	 	.attr("height", m1Variables.sexButtonHeight).attr("width", m1Variables.sexButtonWidth)
	 	.attr("rx", 5).attr("ry", 5).style("opacity", 0)
     		.on('mousedown', m1clickFButton);

	 m1ButtonGroups.mButton.append("rect")
	 	.attr("height", m1Variables.sexButtonHeight).attr("width", m1Variables.sexButtonWidth)
	 	.attr("rx", 5).attr("ry", 5).attr("fill", m1Parameters.mainColor)

     m1ButtonGroups.mButton.append("text")
		.attr("x", m1Variables.sexButtonWidth/2).attr("y", m1Variables.sexButtonHeight/2 + 5)
		.attr("class", "sexButtonText").style('text-anchor', 'middle').attr("fill", "black")
		.text('\uf183');

	 m1ButtonGroups.mButton.append("rect")
	 	.attr("height", m1Variables.sexButtonHeight).attr("width", m1Variables.sexButtonWidth)
	 	.attr("rx", 5).attr("ry", 5).style("opacity", 0)
     		.on('mousedown',  m1clickMButton);

	 // m1ButtonGroups.eButton.append("rect")
	 // 	.attr("height", m1Variables.sexButtonHeight).attr("width", m1Variables.sexButtonWidth)
	 // 	.attr("rx", 5).attr("ry", 5).attr("fill", "black")

  //    m1ButtonGroups.eButton.append("text")
		// .attr("x", m1Variables.sexButtonWidth/2).attr("y", m1Variables.sexButtonHeight/2 + 5)
		// .attr("class", "sexButtonText").style('text-anchor', 'middle').attr("fill", "#eeeeee")
		// .text('ED');

	 // m1ButtonGroups.eButton.append("rect")
	 // 	.attr("height", m1Variables.sexButtonHeight).attr("width", m1Variables.sexButtonWidth)
	 // 	.attr("rx", 5).attr("ry", 5).style("opacity", 0)
  //    		.on('mousedown', clickEdButton);     	
	m1plotGrp.append("text")
		.attr("x", m1Parameters.plotWidth/2)
		.attr("text-anchor", "middle")
		.attr("class", "plotTitle")
		.attr("y", 0 - 15)
		.text("Marathon Finishing Time")

}



//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//---------------------------------------------BUTTON AND SLIDER CONTROLLERS------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------

function m1clickFMButton(){
	if (m1Parameters.buttonStatus.femalemale == false){
		m1resetSexButtons();
		m1Parameters.buttonStatus.femalemale = true;
		m1ButtonGroups.fmButton.select("rect").attr("fill","black");
		m1ButtonGroups.fmButton.select("text").attr("fill","white");	
		m1maskglobalData();
		m1plotHistogram();				
	}
}

function m1clickFButton(){
	if (m1Parameters.buttonStatus.female == false){
		m1resetSexButtons();
		m1Parameters.buttonStatus.female = true;
		m1ButtonGroups.fButton.select("rect").attr("fill","black");
		m1ButtonGroups.fButton.select("text").attr("fill","white");	
		m1maskglobalData();
		m1plotHistogram();				
	}
}

function m1clickMButton(){
	if (m1Parameters.buttonStatus.male == false){
		m1resetSexButtons();
		m1Parameters.buttonStatus.male = true;
		m1ButtonGroups.mButton.select("rect").attr("fill","black");
		m1ButtonGroups.mButton.select("text").attr("fill","white");	
		m1maskglobalData();
		m1plotHistogram();				
	}
}

function m1resetSexButtons(){
	m1ButtonGroups.fmButton.select("rect").attr("fill", m1Parameters.mainColor);
	m1ButtonGroups.fButton.select("rect").attr("fill", m1Parameters.mainColor);
	m1ButtonGroups.mButton.select("rect").attr("fill", m1Parameters.mainColor);
	m1ButtonGroups.fmButton.select("text").attr("fill", "black");
	m1ButtonGroups.fButton.select("text").attr("fill", "black");
	m1ButtonGroups.mButton.select("text").attr("fill", "black");
	m1Parameters.buttonStatus.male = false;
	m1Parameters.buttonStatus.female = false;
	m1Parameters.buttonStatus.femalemale = false;
}

// function clickEdButton(){
// 	console.log("clicked Ed Button");
// 	m1Parameters.buttonStatus.ed = !m1Parameters.buttonStatus.ed;
// 	if (m1Parameters.buttonStatus.ed){
// 		m1ButtonGroups.eButton.select("rect").attr("fill","black");
// 		m1ButtonGroups.eButton.select("text").attr("fill","white");
// 	} else {
// 		m1ButtonGroups.eButton.select("rect").attr("fill", m1Parameters.mainColor);
// 		m1ButtonGroups.eButton.select("text").attr("fill","black");
// 	}
// 	m1maskglobalData();
// 	m1plotHistogram();		
// }



function m1moveYearSlider(xpos){
	newyear = Math.min(m1Parameters.years[m1Parameters.years.length - 1], Math.max(m1Parameters.years[0], Math.round(m1Scales.xYearSliderInv(xpos))))
	if ((Math.round(m1Scales.xYearSliderInv(xpos)) <= m1Parameters.years[0] - 2)){
		newyear = m1Parameters.years[0] - 2;
	}
	if (newyear != m1Variables.year){
		m1Variables.year = newyear;
		m1Variables.allYears = (m1Variables.year == m1Parameters.years[0] - 2);		
		console.log("Year changed to " + m1Variables.year);
		m1maskglobalData();
		console.log("New m1Variables.maskedGlobalData has size" + m1Variables.maskedGlobalData.length)
		m1plotHistogram();
		d3.select("#m1yearHandle").attr("cx", m1Scales.xYearSlider(m1Variables.year))
	}
}

function m1moveLeftAgeSlider(xpos){
	console.log("moving, xpos = " + xpos + "which is a year of "  );
	roundedAge = Math.min(m1Parameters.ages[m1Parameters.ages.length - 1], Math.max(m1Parameters.ages[0], Math.round(m1Scales.xAgeSliderInv(xpos)/5)*5));
	console.log("rounded age = " + roundedAge)
	if ((roundedAge != m1Variables.ageMin) && (roundedAge < m1Variables.ageMax)){
		m1Variables.ageMin = roundedAge;
		console.log("m1Variables.ageMin changed to"+m1Variables.ageMin)
		m1maskglobalData();
		m1plotHistogram();
		d3.select("#m1ageLeftHandle").attr("transform", "translate("+m1Scales.xAgeSlider(m1Variables.ageMin)+",0)");
		d3.select(".m1selectedAgeTrack").attr("x1", m1Scales.xAgeSlider(m1Variables.ageMin));
	}
}

function m1moveRightAgeSlider(xpos){
	roundedAge = Math.min(m1Parameters.ages[m1Parameters.ages.length - 1], Math.max(m1Parameters.ages[1], Math.round(m1Scales.xAgeSliderInv(xpos)/5)*5));
	if ((roundedAge != m1Variables.ageMin) && (roundedAge > m1Variables.ageMin)){
		m1Variables.ageMax = roundedAge;
		m1maskglobalData();
		m1plotHistogram();
		d3.select("#m1ageRightHandle").attr("transform", "translate("+m1Scales.xAgeSlider(m1Variables.ageMax)+",0)");
		d3.select(".m1selectedAgeTrack").attr("x2", m1Scales.xAgeSlider(m1Variables.ageMax));
	}
}
