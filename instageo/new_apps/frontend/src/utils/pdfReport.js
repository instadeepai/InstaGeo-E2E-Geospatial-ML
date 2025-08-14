import React from 'react';
import ReactDOM from 'react-dom/client';
import { jsPDF } from 'jspdf';
import { Canvg } from 'canvg';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis } from 'recharts';
import { generateSegmentationColors } from '../components/../utils/segmentationColors';

// Viridis palette for gradients & bar colors
const VIRIDIS_PALETTE = ['#440154','#482777','#3f4a8a','#31678e','#26838f','#1f9d8a','#6cce5a','#b6de2b','#fee825'];

// Util to fetch remote image -> dataURL
async function fetchImageDataURL(url) {
  const res = await fetch(url);
  const blob = await res.blob();
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.readAsDataURL(blob);
  });
}

// Generic helper: render arbitrary JSX (Recharts chart) to PNG DataURL
async function renderChartToPNG(jsxElement, width, height) {
  const container = document.createElement('div');
  container.style.width = `${width}px`;
  container.style.height = `${height}px`;
  document.body.appendChild(container);

  let root = null;
  try {
    root = ReactDOM.createRoot(container);
    await new Promise((resolve) => {
      root.render(
        <React.Profiler id="chart" onRender={resolve}>
          {jsxElement}
        </React.Profiler>
      );
    });

    const svg = container.querySelector('svg');
    if (!svg) return null;
    svg.setAttribute('width', width.toString());
    svg.setAttribute('height', height.toString());
    svg.removeAttribute('style');

    const svgString = new XMLSerializer().serializeToString(svg);
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const v = await Canvg.fromString(ctx, svgString);
    await v.render();
    return canvas.toDataURL('image/png');
  } finally {
    if (root) root.unmount();
    container.remove();
  }
}

// ------------------- Pie Chart -------------------
export async function generatePieChartPNG(classIndices, proportions, classColors, size = 600) {
  const data = classIndices.map(idx => ({
    value: Number(proportions[idx] || 0),
    name: idx,
    fill: classColors[idx]
  })).filter(d => d.value > 0);
  if (!data.length) return null;

  const chartJSX = (
    <PieChart width={size} height={size} id="pieChart">
      <Pie
        data={data}
        dataKey="value"
        nameKey="name"
        cx={size / 2}
        cy={size / 2}
        innerRadius={0}
        outerRadius={size * 0.45}
        paddingAngle={0}
        startAngle={90}
        endAngle={-270}
        isAnimationActive={false}
        label={({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
          const RAD = Math.PI / 180;
          const radius = innerRadius + (outerRadius - innerRadius) / 2;
          const x = cx + radius * Math.cos(-midAngle * RAD);
          const y = cy + radius * Math.sin(-midAngle * RAD);
          return percent > 0.05 ? (
            <text
              x={x}
              y={y}
              fill="white"
              textAnchor="middle"
              dominantBaseline="central"
              style={{ fontSize: '24px', fontWeight: 'bold', pointerEvents: 'none' }}
            >{`${Math.round(percent * 100)}%`}</text>
          ) : null;
        }}
        labelLine={false}
      >
        {data.map((e, i) => (
          <Cell key={`c-${i}`} fill={e.fill} />
        ))}
      </Pie>
    </PieChart>
  );

  return await renderChartToPNG(chartJSX, size, size);
}

// ------------------- Histogram -------------------
export async function generateHistogramPNG(counts=[], bins=[], size=800, height=600, asPercent=false){
  if(!counts.length||!bins.length) return null;
  const data = counts.map((c,i)=>({ value:c, bin: (bins[i]+bins[i+1])/2 }));
  const interpColor=(t)=>{const n=VIRIDIS_PALETTE.length-1;const idx=Math.floor(t*n);const frac=t*n-idx;const h=(hex)=>[parseInt(hex.substr(1,2),16),parseInt(hex.substr(3,2),16),parseInt(hex.substr(5,2),16)];const a=h(VIRIDIS_PALETTE[idx]);const b=h(VIRIDIS_PALETTE[idx+1]);const rgb=a.map((v,i)=>Math.round(v+(b[i]-v)*frac));return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;};

  const chartJSX = (
    <BarChart width={size} height={height} data={data} margin={{ top: 20, right: 30, left: 40, bottom: 20 }}>
      <XAxis dataKey="bin" tickFormatter={(v) => v.toFixed(2)} tick={{ fontSize: 16 }} />
      <YAxis tickFormatter={(v) => (asPercent ? `${v.toFixed(1)}%` : v.toFixed(0))} tick={{ fontSize: 16 }} />
      <Bar dataKey="value" isAnimationActive={false}>
        {data.map((e, i) => {
          const t = (e.bin - bins[0]) / (bins[bins.length - 1] - bins[0]);
          return <Cell key={i} fill={interpColor(t)} />;
        })}
      </Bar>
    </BarChart>
  );

  return await renderChartToPNG(chartJSX, size, height);
}

// ------------------- Main PDF generator -------------------
export async function generateTaskPdf(taskLayer) {
  try {
    const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();

    // ---------- Banner ---------- //
    const headerH = 18;
    doc.setFillColor(33, 150, 243);
    doc.rect(0, 0, pageWidth, headerH, 'F');

    // Logo (left)
    try {
      const logoUrl = 'https://raw.githubusercontent.com/instadeepai/InstaGeo-E2E-Geospatial-ML/0910341cef9a858137f4dffc1467de7b3240ec0f/assets/logo-dark.png';
      const logoData = await fetchImageDataURL(logoUrl);
      if (logoData) {
        const desiredW = 30;
        const img = new Image();
        const loaded = new Promise((res) => (img.onload = res));
        img.src = logoData;
        await loaded;
        const logoH = desiredW * (img.height / img.width);
        doc.addImage(logoData, 'PNG', 4, 4, desiredW, logoH);
      }
    } catch (e) {
      console.warn('logo load failed', e);
    }

    doc.setTextColor(255, 255, 255);
    doc.setFontSize(14);
    doc.text('Task Report', pageWidth / 2, 7, { align: 'center' });
    doc.setFontSize(10);
    doc.text(`${taskLayer.taskName || taskLayer.id} • ${new Date().toLocaleString()}`, pageWidth / 2, 14, { align: 'center' });

    doc.setTextColor(0, 0, 0);
    let currentY = headerH + 4;

    const drawSectionHeader = (title) => {
      doc.setFillColor(240, 240, 240);
      doc.setDrawColor(200, 200, 200);
      doc.rect(10, currentY, pageWidth - 20, 8, 'FD');
      doc.setFontSize(11);
      doc.text(title, pageWidth / 2, currentY + 5, { align: 'center' });
      currentY += 10;
    };

    // ---------- Satellite & Prediction Images ---------- //
    const satData = taskLayer.satellitePreviewUrl ? await fetchImageDataURL(taskLayer.satellitePreviewUrl) : null;
    const predData = taskLayer.predictionPreviewUrl ? await fetchImageDataURL(taskLayer.predictionPreviewUrl) : null;
    if (satData || predData) {
      if (currentY + 100 > 280) { doc.addPage(); currentY = 10; }
      drawSectionHeader('Satellite RGB Composite & Prediction Images');
      if (satData) doc.addImage(satData, 'JPEG', 10, currentY, 90, 90);
      if (predData) doc.addImage(predData, 'JPEG', 110, currentY, 90, 90);
      currentY += 120; // include imagery height + spacer
    }

    // ---------- Results Overview ---------- //
    if (currentY + 100 > 280) { doc.addPage(); currentY = 10; }
    drawSectionHeader(`${taskLayer.modelType.toUpperCase()} Results Overview on Valid Pixels`);

    if (taskLayer.predictionStats?.type === 'seg') {
      // ---- Segmentation ---- //
      const stats = taskLayer.predictionStats;
      const classColors = generateSegmentationColors(stats.class_indices);
      const pieSize = 600;
      const piePdfSize = 90;
      const piePNG = await generatePieChartPNG(stats.class_indices, stats.class_proportions, classColors, pieSize);
      if (piePNG) doc.addImage(piePNG, 'PNG', 10, currentY, piePdfSize, piePdfSize);

      // legend
      const legendX = piePdfSize + 20;
      let legendY = currentY + 2;
      stats.class_indices.forEach((idx) => {
        const name = stats.classes_mapping?.[idx] || `Class ${idx}`;
        const hex = classColors[idx];
        const [r, g, b] = [parseInt(hex.substr(1,2),16), parseInt(hex.substr(3,2),16), parseInt(hex.substr(5,2),16)];
        doc.setFillColor(r, g, b);
        doc.rect(legendX, legendY, 4, 4, 'F');
        doc.setFontSize(10);
        doc.setTextColor(0,0,0);
        doc.text(name, legendX + 6, legendY + 3);
        legendY += 6;
      });
      currentY += piePdfSize + 4;
    } else if (taskLayer.predictionStats?.type === 'reg') {
      // ---- Regression ---- //
      const stats = taskLayer.predictionStats;
      const counts = stats.histogram?.[0] || [];
      const bins = stats.histogram?.[1] || [];
      const totalPixels = stats.valid_pixels || counts.reduce((a,b)=>a+b,0);
      const percCounts = counts.map(c => totalPixels ? (c/totalPixels*100) : c);
      const histPNG = await generateHistogramPNG(percCounts, bins, 800, 600, true);
      if (histPNG) {
        doc.setFontSize(10);
        // doc.text('Distribution of Value Intervals (%)', 10, currentY);
        // currentY += 4;
        const histW = 120; const histH = 90;
        const histX = 10, histY = currentY;
        doc.addImage(histPNG, 'PNG', histX, histY, histW, histH);

        // legend to right
        const legendX = histX + histW + 15;
        let legendY = histY + 17;
        const hexToRgb = (h) => [parseInt(h.substr(1,2),16),parseInt(h.substr(3,2),16),parseInt(h.substr(5,2),16)];
        const rangeMin = bins[0];
        const rangeMax = bins[bins.length-1];
        const interp = (t)=>{const n=VIRIDIS_PALETTE.length-1;const i=Math.floor(t*n);const f=t*n-i;const a=hexToRgb(VIRIDIS_PALETTE[i]);const b=hexToRgb(VIRIDIS_PALETTE[i+1]);return a.map((v,idx)=>Math.round(v+(b[idx]-v)*f));};
        bins.slice(0,-1).forEach((start,i)=>{
          const end=bins[i+1];
          const center=(start+end)/2;
          const t=(center-rangeMin)/(rangeMax-rangeMin);
          const [r,g,b]=interp(t);
          doc.setFillColor(r,g,b);
          doc.rect(legendX, legendY, 4,4,'F');
          doc.setFontSize(7);
          doc.setTextColor(0,0,0);
          doc.text(`${start.toFixed(3)} - ${end.toFixed(3)}`, legendX+6, legendY+3);
          legendY+=6;
        });
        currentY += histH + 4;
      }
    }

    // ---------- Footer ---------- //
    const totalPages = doc.internal.getNumberOfPages();
    for(let p=1;p<=totalPages;p++){
      doc.setPage(p);
      doc.setFontSize(8);
      doc.setTextColor(150,150,150);
      doc.text(`Page ${p} of ${totalPages}`, pageWidth/2, pageHeight-5, {align:'center'});
    }

    doc.save(`${taskLayer.taskName || taskLayer.id}_report.pdf`);
  } catch (e) {
    console.error('PDF generation error', e);
  }
}
