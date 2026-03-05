// --- GLOBAL STATE ---
let dataset = [];
let metadata = {};
let colorMode = 'original';

// D3 Brushes
const brushPCA = d3.brush();
const brushMDS = d3.brush();

// --- DISCRETE COLOR SCALES (ColorBrewer 5-class) ---
const colorOriginal = d3.scaleOrdinal(d3.schemeCategory10);

// Precision (Blues): 0 = Bad (Dark Blue), 1 = Good (Light Blue). Array reversed so low values get dark colors.
const bluesDiscrete = ["#08519c", "#3182bd", "#6baed6", "#bdd7e7", "#eff3ff"].reverse();
const colorPrecision = d3.scaleQuantize().domain([0, 1]).range(bluesDiscrete); 

// Recall (Reds): 0 = Bad (Dark Red), 1 = Good (Light Red). 
const redsDiscrete = ["#a50f15", "#de2d26", "#fb6a4a", "#fcae91", "#fee5d9"].reverse();
const colorRecall = d3.scaleQuantize().domain([0, 1]).range(redsDiscrete);

// F-Score (RdYlGn): 0 = Bad (Red), 0.5 = Yellow, 1 = Good (Green)
const rdYlGnDiscrete = ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"];
const colorFScore = d3.scaleQuantize().domain([0, 1]).range(rdYlGnDiscrete);

// --- INITIALIZATION ---
d3.json("../json/step2_final_data.json").then(data => {
    dataset = data.points;
    metadata = data.metadata;

    d3.select("#global-assessment").html(`
        Dataset: ${metadata.dataset} <br>
        <span style="color: #2980b9;">${metadata.global_assessment.message}</span>
    `);

    // Draw Scatter Plots
    drawPlot("#pca-plot", "pca_x", "pca_y", "pca", brushPCA);
    drawPlot("#mds-plot", "mds_x", "mds_y", "mds", brushMDS);
    
    // Configure Bi-directional Brushing
    setupBrushing();
    
    // Setup initial empty gauge
    initGauge();

    // Listeners for Radio buttons
    d3.selectAll("input[name='colorMode']").on("change", function() {
        colorMode = this.value;
        updateColors();
        updateLegend();
    });

    updateLegend();
}).catch(err => console.error("Error loading JSON:", err));


// --- PLOTTING FUNCTION ---
function drawPlot(containerSelector, xKey, yKey, plotId, brushObj) {
    const container = d3.select(containerSelector);
    const width = container.node().clientWidth;
    const height = container.node().clientHeight;
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };

    const svg = container.append("svg")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const xScale = d3.scaleLinear().domain(d3.extent(dataset, d => d[xKey])).nice().range([0, innerWidth]);
    const yScale = d3.scaleLinear().domain(d3.extent(dataset, d => d[yKey])).nice().range([innerHeight, 0]);

    svg.append("g").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(xScale));
    svg.append("g").call(d3.axisLeft(yScale));

    // Append brush group BEFORE circles so dots are clickable/hoverable on top
    const brushGroup = svg.append("g").attr("class", "brush-group");

    svg.selectAll(".dot")
        .data(dataset)
        .enter().append("circle")
        .attr("class", `dot dot-${plotId}`)
        .attr("id", d => `dot-${d.id}`)
        .attr("cx", d => xScale(d[xKey]))
        .attr("cy", d => yScale(d[yKey]))
        .attr("r", 5)
        .attr("fill", d => getColor(d))
        .attr("stroke", "#333")
        .attr("stroke-width", 0.5)
        .attr("opacity", 0.8)
        .style("cursor", "pointer")
        // Hover
        .on("mouseover", function(event, d) {
            d3.selectAll(`#dot-${d.id}`).attr("r", 8).attr("stroke", "black").attr("stroke-width", 2);
            showTooltip(event, d);
        })
        .on("mouseout", function(event, d) {
            d3.selectAll(`#dot-${d.id}`).attr("r", 5).attr("stroke", "#333").attr("stroke-width", 0.5);
            hideTooltip();
        })
        // Single Click Selection
        .on("click", function(event, d) {
            // Clear brushes
            d3.select("#pca-plot .brush-group").call(brushPCA.move, null);
            d3.select("#mds-plot .brush-group").call(brushMDS.move, null);
            
            // Isolate point
            d3.selectAll("circle.dot").attr("opacity", 0.1);
            d3.selectAll(`#dot-${d.id}`).attr("opacity", 1);
            
            updateLiveAnalytics([d]);
        });

    // Attach brush
    brushObj.extent([[0, 0], [innerWidth, innerHeight]]);
    brushGroup.call(brushObj);
    
    // Save scales to the brush object for coordinate calculation
    brushObj.xScale = xScale;
    brushObj.yScale = yScale;
}

// --- BI-DIRECTIONAL BRUSHING ---
function setupBrushing() {
    brushPCA.on("start brush end", function(event) {
        if(event.sourceEvent && event.sourceEvent.type === "mousedown") {
            // Clear the other brush to avoid conflict
            d3.select("#mds-plot .brush-group").call(brushMDS.move, null);
        }
        handleBrush(event, brushPCA, "pca_x", "pca_y");
    });

    brushMDS.on("start brush end", function(event) {
        if(event.sourceEvent && event.sourceEvent.type === "mousedown") {
            // Clear the other brush to avoid conflict
            d3.select("#pca-plot .brush-group").call(brushPCA.move, null);
        }
        handleBrush(event, brushMDS, "mds_x", "mds_y");
    });
}

function handleBrush(event, brushObj, xKey, yKey) {
    if (!event.selection) {
        if (event.type === "end") { // Only reset opacities if brush is completely cleared
            d3.selectAll("circle.dot").attr("opacity", 0.8);
            d3.select("#brush-stats").classed("hidden", true);
        }
        return;
    }

    const [[x0, y0], [x1, y1]] = event.selection;
    let selectedPoints = [];

    // Evaluate selection
    dataset.forEach(d => {
        const cx = brushObj.xScale(d[xKey]);
        const cy = brushObj.yScale(d[yKey]);
        const isSelected = x0 <= cx && cx <= x1 && y0 <= cy && cy <= y1;
        
        if (isSelected) selectedPoints.push(d);
        d3.selectAll(`#dot-${d.id}`).attr("opacity", isSelected ? 1 : 0.1);
    });

    updateLiveAnalytics(selectedPoints);
}

// --- COLOR SCALES & LEGEND ---
function getColor(d) {
    if (colorMode === 'original') return colorOriginal(d.label);
    if (colorMode === 'precision') return colorPrecision(d.precision);
    if (colorMode === 'recall') return colorRecall(d.recall);
    if (colorMode === 'fscore') return colorFScore(d.f_score);
}

function updateColors() {
    d3.selectAll("circle.dot").transition().duration(500).attr("fill", d => getColor(d));
}

function updateLegend() {
    const gradient = d3.select("#legend-gradient");
    const desc = d3.select("#legend-description");
    const minLabel = d3.select("#legend-min");
    const maxLabel = d3.select("#legend-max");
    
    gradient.selectAll("*").remove(); // Clear previous blocks

    let colors = [];
    if (colorMode === 'original') {
        gradient.style("background", "linear-gradient(to right, #1f77b4, #ff7f0e, #2ca02c, #d62728)");
        minLabel.text("Class 1"); maxLabel.text("Class N");
        desc.text("Standard categorical colors.");
    } else {
        gradient.style("background", "none");
        if (colorMode === 'precision') { colors = bluesDiscrete; minLabel.text("0 (High FP)"); maxLabel.text("1 (Pure)"); desc.text("5-Class discrete scale for False Positives."); }
        if (colorMode === 'recall') { colors = redsDiscrete; minLabel.text("0 (High FN)"); maxLabel.text("1 (Cohesive)"); desc.text("5-Class discrete scale for False Negatives."); }
        if (colorMode === 'fscore') { colors = rdYlGnDiscrete; minLabel.text("0 (Bad)"); maxLabel.text("1 (Perfect)"); desc.text("5-Class RdYlGn discrete scale for overall structure."); }
        
        // Draw discrete blocks
        colors.forEach(c => {
            gradient.append("div").attr("class", "legend-color-block").style("background-color", c);
        });
    }
}

// --- TACHOMETER (GAUGE CHART) ---
let gaugeForeground;
const gaugeAngleScale = d3.scaleLinear().domain([0, 1]).range([-Math.PI / 2, Math.PI / 2]);

function initGauge() {
    const svg = d3.select("#gauge-chart");
    const width = 160, height = 100;
    const g = svg.append("g").attr("transform", `translate(${width/2},${height - 10})`);

    const arcBg = d3.arc().innerRadius(50).outerRadius(70).startAngle(-Math.PI / 2).endAngle(Math.PI / 2);
    g.append("path").attr("d", arcBg).attr("fill", "#e0e0e0");

    const arcFg = d3.arc().innerRadius(50).outerRadius(70).startAngle(-Math.PI / 2).cornerRadius(3);
    gaugeForeground = g.append("path").datum({ endAngle: -Math.PI / 2 }).attr("fill", "#bdc3c7").attr("d", arcFg);
}

function updateLiveAnalytics(selectedPoints) {
    const statsBox = d3.select("#brush-stats");
    if (selectedPoints.length === 0) {
        statsBox.classed("hidden", true);
        return;
    }
    
    statsBox.classed("hidden", false);
    d3.select("#stat-count").text(selectedPoints.length);
    
    const avgPrec = d3.mean(selectedPoints, d => d.precision);
    const avgRecall = d3.mean(selectedPoints, d => d.recall);
    const avgFScore = d3.mean(selectedPoints, d => d.f_score);

    d3.select("#stat-precision").text(avgPrec.toFixed(3));
    d3.select("#stat-recall").text(avgRecall.toFixed(3));
    
    // Update Gauge
    d3.select("#gauge-value").text(avgFScore.toFixed(3));
    
    const arcFg = d3.arc().innerRadius(50).outerRadius(70).startAngle(-Math.PI / 2).cornerRadius(3);
    const targetAngle = gaugeAngleScale(avgFScore);
    const targetColor = colorFScore(avgFScore); // Color gauge based on the discrete F-Score palette

    gaugeForeground.transition().duration(750)
        .attrTween("d", function(d) {
            const interpolate = d3.interpolate(d.endAngle, targetAngle);
            return function(t) {
                d.endAngle = interpolate(t);
                return arcFg(d);
            };
        })
        .attr("fill", targetColor);
}

// --- TOOLTIP ---
function showTooltip(event, d) {
    const tooltip = d3.select("#tooltip");
    tooltip.transition().duration(100).style("opacity", 1);
    tooltip.html(`
        <strong>ID:</strong> ${d.id} | <strong>Class:</strong> ${d.label}<br>
        <strong>Precision:</strong> ${d.precision.toFixed(3)}<br>
        <strong>Recall:</strong> ${d.recall.toFixed(3)}<br>
        <strong>F-Score:</strong> ${d.f_score.toFixed(3)}
    `)
    .style("left", (event.pageX + 15) + "px")
    .style("top", (event.pageY - 28) + "px");
}

function hideTooltip() {
    d3.select("#tooltip").transition().duration(200).style("opacity", 0);
}