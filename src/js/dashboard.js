// --- GLOBAL STATE ---
let dataset = [];
let metadata = {};
let colorMode = 'original';
let uniqueClasses = [];
let pointById = new Map();

// D3 Brushes
const brushPCA = d3.brush();
const brushMDS = d3.brush();

// --- DISCRETE COLOR SCALES OTTIMIZZATE ---
// Originali: d3.schemeTableau10, ma con il rosso sostituito da un verde per coerenza tematica.
const customTableau = [...d3.schemeTableau10];
customTableau[2] = '#2ca02c'; // Sostituisce il rosso (#e15759) con un verde standard.
const colorOriginal = d3.scaleOrdinal(customTableau);

// Precision (Blues): Scala migliorata e più intuitiva. 0 (Poco preciso) = Azzurro chiaro -> 1 (Molto preciso) = Blu scuro.
const bluesDiscrete = ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"];
const colorPrecision = d3.scaleQuantize().domain([0, 1]).range(bluesDiscrete); 

// Recall (Reds): Scala migliorata e più intuitiva. 0 (Poco coeso) = Rosa chiaro -> 1 (Molto coeso) = Rosso scuro.
const redsDiscrete = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"];
const colorRecall = d3.scaleQuantize().domain([0, 1]).range(redsDiscrete);

// F-Score (Red-Yellow-Green): Scala più intuitiva. 0 (Pessimo) = Rosso, 0.5 = Giallo, 1 (Ottimo) = Verde.
const rdYlGnDiscrete = ["#d73027", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"];
const colorFScore = d3.scaleQuantize().domain([0, 1]).range(rdYlGnDiscrete);

// Tachometri
const gauges = {
    precision: { foreground: null },
    recall: { foreground: null },
    fscore: { foreground: null }
};
const gaugeAngleScale = d3.scaleLinear().domain([0, 1]).range([-Math.PI / 2, Math.PI / 2]);

// --- INITIALIZATION ---
d3.json("../json/step2_final_data.json?v=" + Date.now()).then(data => {
    dataset = data.points;
    metadata = data.metadata;
    uniqueClasses = Array.from(new Set(dataset.map(d => d.label))).sort();
    pointById = new Map(dataset.map(p => [p.id, p]));

    // 1. Inject Global Assessment
    d3.select("#global-assessment").html(`
        Dataset: <strong>${metadata.dataset}</strong> <br>
        <span style="color: #2980b9;">${metadata.global_assessment.message}</span>
    `);

    // 2. Inject Static Analytics (Trustworthiness & Continuity) into Right Panel
    if(metadata.global_assessment.pca && metadata.global_assessment.mds) {
        const qualityContainer = d3.select(".right-panel").insert("div", "h3")
            .attr("class", "stat-card");

        const pca_trust = (metadata.global_assessment.pca.trustworthiness * 100).toFixed(1);
        const pca_cont = (metadata.global_assessment.pca.continuity * 100).toFixed(1);
        const mds_trust = (metadata.global_assessment.mds.trustworthiness * 100).toFixed(1);
        const mds_cont = (metadata.global_assessment.mds.continuity * 100).toFixed(1);

        qualityContainer.html(`
            <h4 style="margin-bottom: 15px;">PROJECTION QUALITY (GLOBAL)</h4>
            <div class="quality-grid">
                <div class="quality-header"></div>
                <div class="quality-header">Trustworthiness</div>
                <div class="quality-header">Continuity</div>

                <div class="quality-label">PCA</div>
                <div class="quality-value">${pca_trust}%</div>
                <div class="quality-value">${pca_cont}%</div>

                <div class="quality-label">MDS</div>
                <div class="quality-value">${mds_trust}%</div>
                <div class="quality-value">${mds_cont}%</div>
            </div>
        `);
    }

    drawPlot("#pca-plot", "pca_x", "pca_y", "pca", brushPCA);
    drawPlot("#mds-plot", "mds_x", "mds_y", "mds", brushMDS);
    setupBrushing();
    
    initGauge("#gauge-precision", gauges.precision);
    initGauge("#gauge-recall", gauges.recall);
    initGauge("#gauge-fscore", gauges.fscore);

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
    const margin = { top: 35, right: 35, bottom: 35, left: 35 };

    const svgRoot = container.append("svg")
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet")
        .style("width", "100%")
        .style("height", "100%");

    // Aggiungo un listener sull'elemento SVG radice. Questo agisce come un "failsafe"
    // per gli hover "appiccicosi". Se il mouse lascia l'area del grafico rapidamente,
    // l'evento "mouseout" del cerchio potrebbe non attivarsi. Questo garantisce il reset.
    svgRoot.on("mouseleave", resetAllHovers);

    const svg = svgRoot.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const xScale = d3.scaleLinear().domain(d3.extent(dataset, d => d[xKey])).nice().range([0, innerWidth]);
    const yScale = d3.scaleLinear().domain(d3.extent(dataset, d => d[yKey])).nice().range([innerHeight, 0]);

    svg.append("g").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(xScale).ticks(5));
    svg.append("g").call(d3.axisLeft(yScale).ticks(5));

    svg.append("g").attr("class", "link-group");
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
        // FIX 1: Bordo grigio scuro semitrasparente. Ancora visivamente il punto allo sfondo!
        .attr("stroke", "rgba(0,0,0,0.4)")
        .attr("stroke-width", 0.8)
        // FIX 2: Opacità base alzata a 0.9 per colori più vividi
        .attr("opacity", 0.9)
        .style("cursor", "pointer")
        .on("mouseover", function(event, d) {
            // Chiamata preventiva per pulire qualsiasi stato "appiccicoso".
            // Questo risolve il problema dei punti che rimangono evidenziati
            // durante movimenti veloci del mouse DENTRO al grafico, garantendo
            // che solo il punto corrente sia evidenziato.
            resetAllHovers();

            d3.selectAll(`#dot-${d.id}`)
              .attr("r", 8)
              .attr("stroke", "rgba(0,0,0,0.9)") // Bordo più netto all'hover
              .attr("stroke-width", 2)
              .raise();
            showTooltip(event, d);
        })
        .on("mouseout", function(event, d) {
            // Quando il mouse esce da un punto (verso uno spazio vuoto), resetta lo stato.
            // La combinazione di questo e del reset su mouseover garantisce la massima robustezza.
            resetAllHovers();
        })
        .on("click", function(event, d) {
            event.stopPropagation(); 
            updateSelection(d);
        });

    brushObj.extent([[0, 0], [innerWidth, innerHeight]]);
    brushGroup.call(brushObj);
    
    brushObj.xScale = xScale;
    brushObj.yScale = yScale;
}

// --- LOGICA DI CLICK E DISEGNO LINEE ---
function updateSelection(d) {
    d3.select("#pca-plot .brush-group").call(brushPCA.move, null);
    d3.select("#mds-plot .brush-group").call(brushMDS.move, null);

    const neighbors = d.neighbors || [];
    const activeIds = new Set([d.id, ...neighbors]);

    // Modifica opacità: isola punto cliccato e vicini (alzato a 0.15)
    d3.selectAll("circle.dot").attr("opacity", p => activeIds.has(p.id) ? 0.9 : 0.15);

    drawLines("pca", d, neighbors, "pca_x", "pca_y", brushPCA);
    drawLines("mds", d, neighbors, "mds_x", "mds_y", brushMDS);

    updateLiveAnalytics([d]);
}

function drawLines(plotId, sourceD, neighborIds, xKey, yKey, scales) {
    const linkGroup = d3.select(`#${plotId}-plot .link-group`);
    linkGroup.selectAll("line").remove(); 

    const sourceX = scales.xScale(sourceD[xKey]);
    const sourceY = scales.yScale(sourceD[yKey]);

    const linesData = neighborIds.map(id => pointById.get(id)).filter(Boolean);

    linkGroup.selectAll("line")
        .data(linesData)
        .enter().append("line")
        .attr("x1", sourceX)
        .attr("y1", sourceY)
        .attr("x2", target => scales.xScale(target[xKey]))
        .attr("y2", target => scales.yScale(target[yKey]))
        .attr("stroke", target => target.label === sourceD.label ? "#2ca02c" : "#e74c3c")
        .attr("stroke-width", 1.5)
        .attr("opacity", 0.6);
}

// --- BI-DIRECTIONAL BRUSHING ---
function setupBrushing() {
    brushPCA.on("start brush end", function(event) {
        if(event.sourceEvent && event.sourceEvent.type === "mousedown") d3.select("#mds-plot .brush-group").call(brushMDS.move, null);
        handleBrush(event, brushPCA, "pca_x", "pca_y");
    });

    brushMDS.on("start brush end", function(event) {
        if(event.sourceEvent && event.sourceEvent.type === "mousedown") d3.select("#pca-plot .brush-group").call(brushPCA.move, null);
        handleBrush(event, brushMDS, "mds_x", "mds_y");
    });
}

function handleBrush(event, brushObj, xKey, yKey) {
    if (!event.sourceEvent) return;
    d3.selectAll(".link-group line").remove(); 

    if (!event.selection) {
        if (event.type === "end") {
            d3.selectAll("circle.dot").attr("opacity", 0.9); // Torna vivido
            resetAllHovers(); // Rimuove anche eventuali hover rimasti attivi
            d3.select("#brush-stats").classed("hidden", true);
            d3.select("#matrix-container").classed("hidden", true);
        }
        return;
    }

    const [[x0, y0], [x1, y1]] = event.selection;
    let selectedPoints = [];
    dataset.forEach(d => {
        const cx = brushObj.xScale(d[xKey]);
        const cy = brushObj.yScale(d[yKey]);
        const isSelected = x0 <= cx && cx <= x1 && y0 <= cy && cy <= y1;
        if (isSelected) selectedPoints.push(d);
        d3.selectAll(`#dot-${d.id}`).attr("opacity", isSelected ? 0.9 : 0.15);
    });
    updateLiveAnalytics(selectedPoints);
}

// --- HELPERS ---

/**
 * Resetta lo stato di hover (raggio e bordo) di tutti i punti su entrambi i grafici.
 * Nasconde anche il tooltip. Funziona come un failsafe globale.
 */
function resetAllHovers() {
    d3.selectAll("circle.dot")
        .attr("r", 5)
        .attr("stroke", "rgba(0,0,0,0.4)")
        .attr("stroke-width", 0.8);
    hideTooltip();
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
    const labelsDiv = d3.select("#legend-labels");
    
    gradient.selectAll("*").remove(); 
    labelsDiv.selectAll("*").remove();

    if (colorMode === 'original') {
        gradient.style("border", "none").style("background", "transparent").style("justify-content", "flex-end").style("gap", "15px");
        uniqueClasses.forEach(cls => {
            const item = gradient.append("div").attr("class", "legend-cluster-item");
            item.append("div").attr("class", "legend-cluster-dot").style("background-color", colorOriginal(cls));
            item.append("span").text(`Class ${cls}`);
        });
    } else {
        gradient.style("border", "1px solid #ccc").style("gap", "0");
        let colors = [];
        let minT = "", maxT = "";

        if (colorMode === 'precision') { colors = bluesDiscrete; minT = "0% (Many FPs)"; maxT = "100% (Pure)"; }
        if (colorMode === 'recall') { colors = redsDiscrete; minT = "0% (Many FNs)"; maxT = "100% (Cohesive)"; }
        if (colorMode === 'fscore') { colors = rdYlGnDiscrete; minT = "0% (Bad)"; maxT = "100% (Perfect)"; }
        
        colors.forEach(c => gradient.append("div").attr("class", "legend-color-block").style("background-color", c));
        labelsDiv.append("span").text(minT);
        labelsDiv.append("span").text(maxT);
    }
}

// --- 3 TACHOMETRI (GAUGES) RESPONSIVE ---
function initGauge(selector, gaugeObj) {
    const svg = d3.select(selector);
    const width = 100, height = 60;

    svg.attr("viewBox", `0 0 ${width} ${height}`)
       .style("width", "100%")
       .style("height", "auto");

    const g = svg.append("g").attr("transform", `translate(${width/2},${height - 5})`);
    const arcBg = d3.arc().innerRadius(30).outerRadius(45).startAngle(-Math.PI / 2).endAngle(Math.PI / 2);
    g.append("path").attr("d", arcBg).attr("fill", "#e0e0e0");

    const arcFg = d3.arc().innerRadius(30).outerRadius(45).startAngle(-Math.PI / 2).cornerRadius(3);
    gaugeObj.foreground = g.append("path").datum({ endAngle: -Math.PI / 2 }).attr("fill", "#bdc3c7").attr("d", arcFg);
}

function updateGauge(gaugeObj, value, color, textSelector) {
    const targetAngle = gaugeAngleScale(value);
    const arcFg = d3.arc().innerRadius(30).outerRadius(45).startAngle(-Math.PI / 2).cornerRadius(3);

    d3.select(textSelector).text((value * 100).toFixed(1) + "%");

    gaugeObj.foreground.transition().duration(750)
        .attrTween("d", function(d) {
            const interpolate = d3.interpolate(d.endAngle, targetAngle);
            return function(t) {
                d.endAngle = interpolate(t);
                return arcFg(d);
            };
        })
        .attr("fill", color);
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

    updateGauge(gauges.precision, avgPrec, colorPrecision(avgPrec), "#val-precision");
    updateGauge(gauges.recall, avgRecall, colorRecall(avgRecall), "#val-recall");
    updateGauge(gauges.fscore, avgFScore, colorFScore(avgFScore), "#val-fscore");
}

// --- TOOLTIP CON DIAGNOSI FP/FN IN INGLESE ---
function showTooltip(event, d) {
    const tooltip = d3.select("#tooltip");
    
    let fpText = d.precision < 0.9 
        ? `🔴 <strong>False Positive:</strong> Attracts <strong>${((1 - d.precision)*100).toFixed(1)}%</strong> of points from other classes.` 
        : `🟢 <strong>Low FPs:</strong> No class mixing.`;
        
    let fnText = d.recall < 0.9 
        ? `🔴 <strong>False Negative:</strong> Disconnected from <strong>${((1 - d.recall)*100).toFixed(1)}%</strong> of points in its own class.` 
        : `🟢 <strong>Low FNs:</strong> Highly cohesive.`;

    tooltip.html(`
        <strong>ID:</strong> ${d.id} | <strong>Class:</strong> ${d.label}<br>
        <strong>Precision:</strong> ${(d.precision*100).toFixed(1)}%<br>
        <strong>Recall:</strong> ${(d.recall*100).toFixed(1)}%<br>
        <strong>F-Score:</strong> ${(d.f_score*100).toFixed(1)}%
        <div class="tt-diagnosis">
            ${fpText}<br>
            ${fnText}
        </div>
    `);

    // --- Logica di posizionamento dinamico per evitare che il tooltip esca dallo schermo ---
    const tooltipNode = tooltip.node();
    const tooltipWidth = tooltipNode.offsetWidth;
    const tooltipHeight = tooltipNode.offsetHeight;
    const margin = 20;

    // Posizione orizzontale: di default a destra, ma a sinistra se non c'è spazio.
    let x = event.pageX + margin;
    if (x + tooltipWidth > window.innerWidth) { x = event.pageX - tooltipWidth - margin; }

    // Posizione verticale: di default sotto, ma sopra se non c'è spazio in basso.
    let y = event.pageY + margin;
    if (y + tooltipHeight > window.innerHeight) { y = event.pageY - tooltipHeight - margin; }

    tooltip.style("left", x + "px").style("top", y + "px").transition().duration(100).style("opacity", 1);
}

function hideTooltip() {
    d3.select("#tooltip").transition().duration(200).style("opacity", 0);
}