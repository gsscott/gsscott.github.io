
function secToString(x){
	hours = ("00" + (Math.floor(x/3600)).toString()).slice(-2)
	minutes = ("00" + (Math.floor(x/60) % 60)).toString().slice(-2)
	seconds = ("00" + (Math.floor(x) % 60)).toString().slice(-2)
	return (hours + ":" + minutes + ":" + seconds)

}

document.querySelector("#calculatorButton").addEventListener("click", function(e){
			console.log("Clicked ");
			var timeInput = document.querySelector("#timeInput").value
			if (timeInput.length > 3){
				seconds = 3600*timeInput.split(":")[0] + 60*timeInput.split(":")[1]
				split5k = 0.115003*seconds
				split10k = 0.229935*seconds
				split15k = 0.345875*seconds
				split20k = 0.462535*seconds
				split211k = 0.487902*seconds
				split25k = 0.579244*seconds
				split30k = 0.698657*seconds
				split35k = 0.821045*seconds
				split40k = 0.945313*seconds
				document.querySelector("#div5k").innerHTML = "5k: <br> " +      secToString(split5k)
				document.querySelector("#div10k").innerHTML = "10k: <br> " +    secToString(split10k)
				document.querySelector("#div15k").innerHTML = "15k: <br> " +    secToString(split15k)
				document.querySelector("#div20k").innerHTML = "20k: <br> " +    secToString(split20k)
				document.querySelector("#div211k").innerHTML = "21.1k: <br> " + secToString(split211k)
				document.querySelector("#div25k").innerHTML = "25k: <br> " +    secToString(split25k)
				document.querySelector("#div30k").innerHTML = "30k: <br> " +    secToString(split30k)
				document.querySelector("#div35k").innerHTML = "35k: <br> " +    secToString(split35k)
				document.querySelector("#div40k").innerHTML = "40k: <br> " +    secToString(split40k)
			}
		})