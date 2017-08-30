//-------------------------------------------PAGE SETUP---------------------------------------------------

const svgWidth=850;
const svgHeight=550;
const margin = {top:25, bottom:125, right:50, left:50};
const sliderTimeMargin = {left:300, right:20}
const sliderAgeMargin = {left:300, right:20}
const plotWidth = svgWidth - margin.right - margin.left;
const plotHeight = svgHeight - margin.top - margin.bottom;
const sliderTimeWidth=svgWidth - margin.left-margin.right-sliderTimeMargin.left - sliderTimeMargin.right;
const sliderAgeWidth=svgWidth - margin.left-margin.right-sliderAgeMargin.left - sliderAgeMargin.right;
const longTransitionTime = 800;
const transitionRatio = 0.8;
const numTicks = 60;
const violinWidth = 60;
const segmentNames = ['0-10km pace (sec/km)', '10-21.1km pace (sec/km)',
       '21.1-30km pace (sec/km)', '30-35km pace (sec/km)',
       '35-40km pace (sec/km)', '40-42.2km pace (sec/km)', 'Pace (sec/km)'];
const shortSegmentNames = ['0-10km pace', '10-21.1km pace','21.1-30km pace','30-35km pace', '35-40km pace','40-42.2km pace','Race pace'];
const tickNames = ['0-10km', '10-21.1km', '21.1-30km', '30-35km', '35-40km', '40-42.2km', 'Entire Race'];
const yMaxLarge = 175;
const yMaxSmall = 50;
const numYTicks = 15;
const mainColor = "lightseagreen";
var transitionTime = 750;
var subtractedSegmentIndex = shortSegmentNames.length - 1;



//------------------------------------------PAREMETER SETUP----------------------------------------------
const years = [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
 				2011, 2012, 2013, 2014, 2015, 2016];
const ages = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]; //15 represents <20, 70 represents >65
const yValues = [10, 30, 100, 300, 1000];
const segmentTicks = [((0 + 0.5)*(plotWidth/7)), ((1 + 0.5)*(plotWidth/7)), ((2 + 0.5)*(plotWidth/7)), ((3 + 0.5)*(plotWidth/7)), ((4 + 0.5)*(plotWidth/7)), ((5 + 0.5)*(plotWidth/7)), ((6 + 0.5)*(plotWidth/7))];
const bgButtonMargin = {left: 25};

var buttonStatus = {mi: false, zoom: false, autozoom: true, femalemale: true, male:false, female:false};
var barStyle = {color: mainColor, opacity:0.8};

const gridAbovePlot = 10;


var sexButtonWidth= 45;
var sexButtonHeight= 25;

var miButtonWidth = 72;
var miButtonHeight = 25;

var zoomButtonWidth= 72;
var zoomButtonHeight= 25;

var cSelectorWidth = 140;
var cSelectorHeight = 25;

var globalData;
var maskedGlobalData;
var maskedGlobalDataNo2012;
var currentBins = [];

var ageMin = ages[0];
var ageMax = ages[ages.length - 1];
var timeMin = 120;
var timeMax = 390;
var yMax = yMaxLarge;






//----------------------------------------APPEND PLOT GROUPS AND CONTROL GROUPS-------------------------
var svg = d3.select("#marathon_plot2")
	.attr("width", svgWidth)
	.attr("height", svgHeight)
	.attr("background", "blue")

var plotGrp = svg.append("g").attr("transform", "translate("+margin.left+","+margin.top+")");
var ctrlGrp = svg.append('g').attr("transform", "translate("+margin.left+","+(svgHeight - margin.bottom + 50)+")");


var yLabel = plotGrp.append("g")
	.attr("text-anchor", "middle")
	.attr("transform", "translate(-35," + plotHeight/2 + ") rotate(-90,0,0)");//translate(0,-200)");

var sliderTimeGrp = ctrlGrp.append('g').attr("transform", "translate("+sliderTimeMargin.left+",0)");
var sliderAgeGrp = ctrlGrp.append('g').attr("transform", "translate("+sliderAgeMargin.left+",50)");

var miButton = ctrlGrp.append('g').attr("transform", "translate("+bgButtonMargin.left+","+(0-(sexButtonHeight/2)) +")");
var zoomButton = ctrlGrp.append('g').attr("transform", "translate("+(bgButtonMargin.left + 10 + miButtonWidth)+","+(0-(sexButtonHeight/2)) +")");
var fmButton = ctrlGrp.append('g').attr("transform", "translate("+bgButtonMargin.left+","+(50-(sexButtonHeight/2)) +")");
var fButton = ctrlGrp.append('g').attr("transform", "translate("+(bgButtonMargin.left+(sexButtonWidth + 10))+","+(50-(sexButtonHeight/2)) +")");
var mButton = ctrlGrp.append('g').attr("transform", "translate("+(bgButtonMargin.left+(2*(sexButtonWidth + 10))) +","+(50-(sexButtonHeight/2)) +")");

var cSelector = ctrlGrp.append('g').attr("transform", "translate(395,"+(-plotHeight - 72)+")");


//----------------------------------------APPEND VIOLIN HOLDERS-----------------------------------------

var violinGrpPlus = [];
for (var i = 0; i < segmentNames.length; i++){
	violinGrpPlus[i] = plotGrp.append("g").attr("transform", "translate(" + (segmentTicks[i]) + ","+plotHeight+") rotate(-90,0,0)");//translate(0,-200)");
}

//-----------------------------------------------SLIDERS----------------------------------------------------------------
var x = d3.scaleLinear().domain([120, 390]).range([0, plotWidth]);
var y = d3.scaleLinear().domain([(-yMax), yMax]).range([plotHeight, 0]);

var xHist = d3.scaleLinear().domain([-(yMax), yMax]).range([0, plotHeight]);
var yHist = d3.scaleLinear().domain([0, 2000]).range([0, (-violinWidth)]);

var xTimeSlider = d3.scaleLinear().domain([120, 390]).range([0, sliderTimeWidth])
var xTimeSliderInv = d3.scaleLinear().domain([0, sliderTimeWidth]).range([120, 390])

var xAgeSlider = d3.scaleLinear().domain([ages[0], ages[ages.length - 1]]).range([0, sliderAgeWidth])
var xAgeSliderInv = d3.scaleLinear().range([ages[0], ages[ages.length - 1]]).domain([0, sliderAgeWidth])

var yAxis = d3.axisLeft(y).tickSize(plotWidth);


d3.csv("https://gsscott.github.io/df2veryslim2002.csv", function(data) {
	globalData = data;
	maskGlobalData();
	//maskedGlobalData = data;
	console.log("Done loading");
	initializeXTicks();
	plotGrp.append('g').attr("class", "yaxis").attr("transform", "translate("+(plotWidth)+",0)").call(yAxis.ticks(numYTicks));
	plotGrp.select(".yaxis").selectAll(".tick line").attr("stroke", "#777").attr("stroke-dasharray", "2,2");
	plotGrp.select(".yaxis").select(".domain").remove();
	yLabel.append("text").text("sec / km");
	plotViolin();
	transitionTime = longTransitionTime;
	initializeControls();
});


//---------------------------------------------BUTTON AND SLIDER CONTROLLERS------------------------------------------
function clickMiButton(){
	buttonStatus.mi = !buttonStatus.mi;
	if (buttonStatus.mi){
		miButton.select("rect").attr("fill","black");
		miButton.select("text").attr("fill","white");	
		yLabel.select("text").text("sec / mi");
		yAxis.tickFormat(d => Math.round(d * 1.60934))	
	}	else {
		miButton.select("rect").attr("fill",mainColor);
		miButton.select("text").attr("fill","black");		
		yAxis.tickFormat(d => d)
		yLabel.select("text").text("sec / km");
	}
	plotGrp.select(".yaxis").call(yAxis.ticks(numYTicks));
	plotGrp.select(".yaxis").select(".domain").remove();
}

function clickZoomButton(){
	buttonStatus.zoom = !buttonStatus.zoom;
	if (buttonStatus.zoom){
		yMax = yMaxSmall
		zoomButton.select("rect").attr("fill","black");
		zoomButton.select("text").attr("fill","white");		
	}	else {
		yMax = yMaxLarge
		zoomButton.select("rect").attr("fill",mainColor);
		zoomButton.select("text").attr("fill","black");		
	}
	y.domain([(-yMax), yMax]);
	xHist.domain([-(yMax), yMax]);
	plotGrp.select(".yaxis").transition().duration(transitionTime).call(yAxis.ticks(numYTicks));
	plotGrp.select(".yaxis").selectAll(".tick line").attr("stroke", "#777").attr("stroke-dasharray", "2,2");
	plotGrp.select(".yaxis").select(".domain").remove();
	var tempTT = transitionTime;
	transitionTime = 0;
	plotViolin();
	transitionTime = tempTT;
}

function clickFMButton(){
	if (buttonStatus.femalemale == false){
		resetSexButtons();
		buttonStatus.femalemale = true;
		fmButton.select("rect").attr("fill","black");
		fmButton.select("text").attr("fill","white");	
		maskGlobalData();
		plotViolin();				
	}
}

function clickFButton(){
	if (buttonStatus.female == false){
		resetSexButtons();
		buttonStatus.female = true;
		fButton.select("rect").attr("fill","black");
		fButton.select("text").attr("fill","white");	
		maskGlobalData();
		plotViolin();				
	}
}

function clickMButton(){
	if (buttonStatus.male == false){
		resetSexButtons();
		buttonStatus.male = true;
		mButton.select("rect").attr("fill","black");
		mButton.select("text").attr("fill","white");	
		maskGlobalData();
		plotViolin();				
	}
}

function resetSexButtons(){
	fmButton.select("rect").attr("fill", mainColor);
	fButton.select("rect").attr("fill", mainColor);
	mButton.select("rect").attr("fill", mainColor);
	fmButton.select("text").attr("fill", "black");
	fButton.select("text").attr("fill", "black");
	mButton.select("text").attr("fill", "black");
	buttonStatus.male = false;
	buttonStatus.female = false;
	buttonStatus.femalemale = false;
}



function moveLeftAgeSlider(xpos){
	roundedAge = Math.min(ages[ages.length - 1], Math.max(ages[0], Math.round(xAgeSliderInv(xpos)/5)*5));
	if ((roundedAge != ageMin) && (roundedAge < ageMax)){
		ageMin = roundedAge;
		console.log("ageMin changed to"+ageMin)
		maskGlobalData();
		plotViolin();
		d3.select("#ageLeftHandle").attr("transform", "translate("+xAgeSlider(ageMin)+",0)");
		sliderAgeGrp.select(".selectedAgeTrack").attr("x1", xAgeSlider(ageMin));
	}
}

function moveRightAgeSlider(xpos){
	roundedAge = Math.min(ages[ages.length - 1], Math.max(ages[1], Math.round(xAgeSliderInv(xpos)/5)*5));
	if ((roundedAge != ageMin) && (roundedAge > ageMin)){
		ageMax = roundedAge;
		maskGlobalData();
		plotViolin();
		d3.select("#ageRightHandle").attr("transform", "translate("+xAgeSlider(ageMax)+",0)");
		sliderAgeGrp.select(".selectedAgeTrack").attr("x2", xAgeSlider(ageMax));
	}
}


function moveLeftTimeSlider(xpos){
	roundedTime = Math.round(Math.min(timeMax - 15, Math.max(120, xTimeSliderInv(xpos))));
	console.log("Left roundedTime = " + roundedTime)
	if ((roundedTime != timeMin)){
		timeMin = roundedTime;
		console.log("timeMin changed to"+timeMin)
		maskGlobalData();
		d3.select("#timeLeftHandle").attr("transform", "translate("+xTimeSlider(timeMin)+",0)");
		d3.select(".selectedTimeTrack").attr("x1", xTimeSlider(timeMin));
		d3.select("#leftTimeText")
			.attr("transform", "translate("+xTimeSlider(timeMin)+",20)")
			.text(minsToTime(timeMin));
	}
}

function moveRightTimeSlider(xpos){
	roundedTime = Math.round(Math.max(timeMin + 15, Math.min(390, xTimeSliderInv(xpos))));
	if ((roundedTime != timeMax)){
		timeMax = roundedTime;
		console.log("timeMax changed to"+timeMax)
		maskGlobalData();
		d3.select("#timeRightHandle").attr("transform", "translate("+xTimeSlider(timeMax)+",0)");
		d3.select(".selectedTimeTrack").attr("x2", xTimeSlider(timeMax));
		d3.select("#rightTimeText")
			.attr("transform", "translate("+xTimeSlider(timeMax)+",20)")
			.text(minsToTime(timeMax));
	}
}



function maskGlobalData(){
	maskedGlobalData = globalData.filter(function(d){ return d['Chip Time (min)'] >= timeMin;})
	maskedGlobalData = maskedGlobalData.filter(function(d){ return d['Chip Time (min)'] <= timeMax;})

	maskedGlobalData = maskedGlobalData.filter(function(d){ return d['Min age'] >= ageMin;})
	maskedGlobalData = maskedGlobalData.filter(function(d){ return d['Max age'] <= ageMax;})
	if (buttonStatus.female){
		maskedGlobalData = maskedGlobalData.filter(function(d){ return (d['Sex'] == "F");})
	}
	if (buttonStatus.male){
		maskedGlobalData = maskedGlobalData.filter(function(d){ return (d['Sex'] == "M");})
	}	
	maskedGlobalDataNo2012 = maskedGlobalData.filter(function(d){ return (d['Year'] != '12');})
}


function plotViolin(){
	var oldYMax = yMax;
	var oldBins = currentBins;
	console.log("In plotviolin");

	var yHistMax = 0;
	for (var segmentNum = 0; segmentNum < segmentNames.length; segmentNum += 1){
		if ((segmentNum != subtractedSegmentIndex) && (shortSegmentNames[segmentNum] == '40-42.2km pace')){
			currentBins[segmentNum] =d3.histogram().domain(xHist.domain()).thresholds(xHist.ticks(numTicks))(maskedGlobalDataNo2012.map(function (d){return parseInt(d[segmentNames[segmentNum]] - d[segmentNames[subtractedSegmentIndex]])}));
			yHistMax = Math.max(yHistMax, d3.max(currentBins[segmentNum], function(d) {return d.length;}));
		} else if (segmentNum != subtractedSegmentIndex) {
			currentBins[segmentNum] =d3.histogram().domain(xHist.domain()).thresholds(xHist.ticks(numTicks))(maskedGlobalData.map(function (d){return parseInt(d[segmentNames[segmentNum]] - d[segmentNames[subtractedSegmentIndex]])}));
			yHistMax = Math.max(yHistMax, d3.max(currentBins[segmentNum], function(d) {return d.length;}));
		}	
	}
	if (buttonStatus.autozoom){
		yHist.domain([0, yHistMax]);
		}

	if (1 == 1){
        var area = d3.area()
            //.curve(d3.curveLinear)
            .curve(d3.curveBasis)
            .x(function(d) {return xHist((d.x0 + d.x1)/2);})
            .y0(function(d) { return (-yHist(d.length)); })	
            .y1(function(d) { return yHist(d.length); });
        var line = d3.line()
            //.curve(d3.curveLinear)
            .curve(d3.curveLinear)
            .x(function(d) {return xHist(d.x0);})
            .y(function(d) { return yHist(d.length); });            

		for (var segmentNum = 0; segmentNum < segmentNames.length; segmentNum++){
			console.log("plotting segmentNum " + segmentNum)
			if (segmentNum != subtractedSegmentIndex) {
	        var v = violinGrpPlus[segmentNum].selectAll(".violin"+segmentNum).data([currentBins[segmentNum]]);
	        var merged = v.enter().append("path")
	        	.attr("class", "area")
	        	.attr("class", "violin"+segmentNum)
	        	.attr("d", area)
	        	.style("fill", mainColor)
	        	.merge(v);
	        merged.transition().duration(transitionTime)
	        	.attr("d", area);
			}

		}




      			          
	}

	}

function initializeXTicks(){
	//----------------------------------------------DRAW THE TICKMARKS-----------------------------------
	for (i = 0; i < (segmentTicks.length); i++){
		console.log("adding line");
		// plotGrp.insert("line", ".medianLine")
		// 	.attr("x1", segmentTicks[i]).attr("x2", segmentTicks[i])
		// 	.attr("y1", plotHeight).attr("y2", plotHeight + 10)
		// 	.attr("class", "majorTimeTick");
		plotGrp.append("text")
			.attr("x", segmentTicks[i])
			.attr("y", plotHeight + 15)
			.attr("class", "majorTimeTickLabel")
			.style('text-anchor', 'middle')
			.text(tickNames[i] );
	}
	plotGrp.append("text")
		.attr("x", plotWidth/2 - (cSelectorWidth/2))
		.attr("text-anchor", "middle")
		.attr("class", "plotTitle")
		.attr("y", 0 - 5)
		.text("Segment pace minus")
}









function initializeControls(){
	//----------------------------------------------DRAW THE 'TIME' SLIDER-------------------------------
	sliderTimeGrp.append("line")
			.attr("class", "trackborder")
			.attr("x1", 0)
			.attr("x2", 0 + sliderTimeWidth)
			.attr("y1", 0).attr("y2", 0);			
	sliderTimeGrp.insert("line", ".trackborder")
			.attr("class", "track")
			.attr("x1", 0)
			.attr("x2", 0 + sliderTimeWidth)
			.attr("y1", 0).attr("y2", 0);
	sliderTimeGrp.append("line")
			.attr("class", "selectedTimeTrack")
			.attr("x1", 0)
			.attr("x2", 0 + sliderTimeWidth)
			.attr("y1", 0).attr("y2", 0);			
	sliderTimeGrp.append("polyline")
			.attr("class", "triangles")
			.attr("id", "timeLeftHandle")
			.attr("transform", "translate("+xTimeSlider(timeMin)+",0)")
			.attr("points", "-7,-12,12,0,-7,12")
			.call(d3.drag()
				//.on("start drag", function() {console.log("mouseup")}));
				.on("start drag", function() {moveLeftTimeSlider(d3.event.x)})
				.on("end", function() {plotViolin()}));
	sliderTimeGrp.append("polyline")
			.attr("class", "triangles")
			.attr("id", "timeRightHandle")
			.attr("transform", "translate("+xTimeSlider(timeMax)+",0)")
			.attr("points", "7,-12,7,12,-12,0")
			.call(d3.drag()
				.on("start drag", function() {moveRightTimeSlider(d3.event.x)})
				.on("end", function() {plotViolin()}));

	sliderTimeGrp.append("text")
		.attr("id", "leftTimeText")
		.attr("text-anchor", "middle")
		.attr("transform", "translate("+xTimeSlider(timeMin)+",20)")
		.style("font-size", "11px")
		.text(minsToTime(timeMin));
	sliderTimeGrp.append("text")
		.attr("id", "rightTimeText")
		.attr("text-anchor", "middle")
		.attr("transform", "translate("+xTimeSlider(timeMax)+",20)")
		.style("font-size", "11px")
		.text(minsToTime(timeMax));		
	sliderTimeGrp.append("text")
			.attr("text-anchor", "end")
			.attr("y", 5).attr("x", -15)
			.text("Finishing time:")		

	//----------------------------------------DRAW THE 'AGE' SLIDER---------------------------------------
	sliderAgeGrp.append("line")
			.attr("class", "trackborder")
			.attr("x1", 0)
			.attr("x2", 0 + sliderAgeWidth)
			.attr("y1", 0).attr("y2", 0);
	sliderAgeGrp.insert("line", ".trackborder")
			.attr("class", "track")
			.attr("x1", 0)
			.attr("x2", 0 + sliderAgeWidth)
			.attr("y1", 0).attr("y2", 0);	
	sliderAgeGrp.insert("line", ".trackborder")
			.attr("class", "selectedAgeTrack")
			.attr("x1", 0)
			.attr("x2", 0 + sliderAgeWidth)
			.attr("y1", 0).attr("y2", 0);		
	for (var i = 0; i < ages.length; i++){
		sliderAgeGrp.insert("circle", ".trackborder")
			.attr("class", "yearCircle")
			.attr("cx", xAgeSlider(ages[i]))
			.attr("r", 6)
		sliderAgeGrp.insert("text")
			.attr("x", xAgeSlider(ages[i]))
			.attr("y", 20).style('text-anchor', 'middle')
			.style("font-size", "11px")
			.text(ages[i].toString());
		}
	sliderAgeGrp.select("text").text("<20");	
	sliderAgeGrp.select("text:last-of-type").text(">70");
	sliderAgeGrp.append("polyline")
			.attr("class", "triangles")
			.attr("id", "ageLeftHandle")
			.attr("transform", "translate("+xAgeSlider(ageMin)+",0)")
			.attr("points", "-7,-12,12,0,-7,12")
			.call(d3.drag()
				.on("start drag", function() {moveLeftAgeSlider(d3.event.x)}));
	sliderAgeGrp.append("polyline")
			.attr("class", "triangles")
			.attr("id", "ageRightHandle")
			.attr("transform", "translate("+xAgeSlider(ageMax)+",0)")
			.attr("points", "7,-12,7,12,-12,0")
			.call(d3.drag()
				.on("start drag", function() {moveRightAgeSlider(d3.event.x)}));
	sliderAgeGrp.append("text")
			.attr("text-anchor", "end")
			.attr("y", 5).attr("x", -15)
			.text("Age:")

	//----------------------------------------------------DRAW BUTTONS--------------------------------------------------



	 miButton.append("rect")
	 	.attr("height", miButtonHeight).attr("width", miButtonWidth)
	 	.attr("rx", 5).attr("ry", 5).attr("fill", mainColor)

     miButton.append("text")
		.attr("x", miButtonWidth/2).attr("y", miButtonHeight/2 + 5)
		.attr("class", "zoomButtonText").style('text-anchor', 'middle').attr("fill", "black")
		.text('Miles');

	 miButton.append("rect")
	 	.attr("height", miButtonHeight).attr("width", miButtonWidth)
	 	.attr("rx", 5).attr("ry", 5).style("opacity", 0)
     		.on('mousedown', clickMiButton);

	 zoomButton.append("rect")
	 	.attr("height", zoomButtonHeight).attr("width", zoomButtonWidth)
	 	.attr("rx", 5).attr("ry", 5).attr("fill", mainColor)

     zoomButton.append("text")
		.attr("x", zoomButtonWidth/2).attr("y", zoomButtonHeight/2 + 5)
		.attr("class", "zoomButtonText").style('text-anchor', 'middle').attr("fill", "black")
		.text('\uf002');

	 zoomButton.append("rect")
	 	.attr("height", zoomButtonHeight).attr("width", zoomButtonWidth)
	 	.attr("rx", 5).attr("ry", 5).style("opacity", 0)
     		.on('mousedown', clickZoomButton);

	 fmButton.append("rect")
	 	.attr("height", sexButtonHeight).attr("width", sexButtonWidth)
	 	.attr("rx", 5).attr("ry", 5).attr("fill", "black")

     fmButton.append("text")
		.attr("x", sexButtonWidth/2).attr("y", sexButtonHeight/2 + 5)
		.attr("class", "sexButtonText").style('text-anchor', 'middle').attr("fill", "#eeeeee")
		.text('\uf182' + ' ' + '\uf183');

	 fmButton.append("rect")
	 	.attr("height", sexButtonHeight).attr("width", sexButtonWidth)
	 	.attr("rx", 5).attr("ry", 5).style("opacity", 0)
     		.on('mousedown', clickFMButton);

	 fButton.append("rect")
	 	.attr("height", sexButtonHeight).attr("width", sexButtonWidth)
	 	.attr("rx", 5).attr("ry", 5).attr("fill", mainColor)

     fButton.append("text")
		.attr("x", sexButtonWidth/2).attr("y", sexButtonHeight/2 + 5)
		.attr("class", "sexButtonText").style('text-anchor', 'middle').attr("fill", 'black')
		.text('\uf182');

	 fButton.append("rect")
	 	.attr("height", sexButtonHeight).attr("width", sexButtonWidth)
	 	.attr("rx", 5).attr("ry", 5).style("opacity", 0)
     		.on('mousedown', clickFButton);

	 mButton.append("rect")
	 	.attr("height", sexButtonHeight).attr("width", sexButtonWidth)
	 	.attr("rx", 5).attr("ry", 5).attr("fill", mainColor)

     mButton.append("text")
		.attr("x", sexButtonWidth/2).attr("y", sexButtonHeight/2 + 5)
		.attr("class", "sexButtonText").style('text-anchor', 'middle').attr("fill", "black")
		.text('\uf183');

	 mButton.append("rect")
	 	.attr("height", sexButtonHeight).attr("width", sexButtonWidth)
	 	.attr("rx", 5).attr("ry", 5).style("opacity", 0)
     		.on('mousedown',  clickMButton);

//----------------------------------------------DRAW THE SELECTOR-------------------------------------
	 cSelector.append("rect")
	 	.attr("height", cSelectorHeight).attr("width", cSelectorWidth)
	 	.attr("rx", 5).attr("ry", 5).attr("fill", mainColor)
	 cSelector.append("text")
		.attr("x", cSelectorWidth/2).attr("y", cSelectorHeight/2 + 5)
		.attr("class", "cSelectorText").style('text-anchor', 'middle').attr("fill", "black")
		.attr("class", "plotTitle")
		.text('Race pace');	 
	 cSelector.on('mouseenter', openMenu)
	 	.on('mouseleave', closeMenu);
	 // cSelector.append("rect")
	 // 	.attr("id", "cSelectorButton")
	 // 	.attr("height", cSelectorHeight).attr("width", cSelectorWidth)
	 // 	.attr("rx", 5).attr("ry", 5).style("opacity", 0)
	 // 		.on('mouseenter', openMenu)
	 // 		.on('mouseleave', closeMenu)
	 // 		.on('mouseout', function() {console.log("mouseout")});
}


function openMenu() {

	optionHeight = 20;
	optionWidth = cSelectorWidth;

	cSelector.append("rect")
		.attr("height", optionHeight * shortSegmentNames.length)
		.attr("width", optionWidth)
		.attr("transform", "translate(0,"+cSelectorHeight+")")
		.attr("fill", "white")
		.attr("class", "menu")
		.attr("border", "1px solid black")
		.style("stroke-width", 1)
		.style("stroke", "black")

	//d3.select("#cSelectorButton").attr("height", cSelectorHeight +  (optionHeight * segmentNames.length));
	for (i = 0; i < shortSegmentNames.length; i++){
		cSelector.insert("text")
			.attr("class", "menu")
			.attr("x", optionWidth / 2)
			.attr("y", cSelectorHeight + (i * optionHeight) + (optionHeight/2) +5)
			.style("text-anchor", "middle").attr("fill", "black")
			.text(shortSegmentNames[i])	
		 cSelector.append("rect")
		 	.attr("class", "menu")
		 	.attr("x", 0)
		 	.attr("y", cSelectorHeight + (i * optionHeight))
		 	.attr("height", optionHeight)
		 	.attr("width", optionWidth)
		 	//.style("stroke-width", 1).style("stroke", "red")
		 	.style("opacity", 0)
		 		.on('mouseenter', function() {d3.select(this).style("opacity", 0.5);})
		 		.on('mouseleave', function() {d3.select(this).style("opacity", 0);})
		 		.on('mousedown', function(i) {return function () {changeSelector(i);}}(i));
	}
}


function closeMenu() {
	console.log("mouseleave")
	d3.selectAll(".menu").transition().duration(100).remove();
}

function minsToTime(x) {
	if (x == 390){return "6:30+"}
	else{return Math.floor(x/60).toString() + ":" + ("00" + (x%60).toString()).slice(-2);}
}

function changeSelector(i) {
	cSelector.select("text").text(shortSegmentNames[i]);
	subtractedSegmentIndex = i;
	d3.selectAll(".violin"+subtractedSegmentIndex).remove();
	plotViolin();
}

