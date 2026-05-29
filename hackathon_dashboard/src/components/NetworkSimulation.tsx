'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Radio, Zap, ShieldAlert, CheckCircle2, Waves } from 'lucide-react';

interface Device {
  id: number;
  x: number;
  y: number;
  label: string;
  phase: number;
}

/**
 * NetworkSimulation: A clean, minimalist graphic demonstrating CSI impact.
 * Visualizes the difference between jittery, degraded paths and precise, AI-optimized beams.
 */
export default function NetworkSimulation() {
  const [isPreciseCSI, setIsPreciseCSI] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    const width = canvas.width = 1200;
    const height = canvas.height = 600;

    const NOKIA_BLUE = '#1241C6';
    const SIGNAL_RED = '#EF4444';
    const GRAPHITE = '#2C2C2C';
    const PAPER_WHITE = '#FDFDFD';

    // Fixed Spatial Distribution (No overlapping)
    const antenna = { x: 200, y: height / 2 };
    const devices: Device[] = [
      { id: 1, x: 850, y: 150, label: "UE-01", phase: 0 },
      { id: 2, x: 1050, y: 300, label: "UE-02", phase: Math.PI / 2 },
      { id: 3, x: 900, y: 480, label: "UE-03", phase: Math.PI },
      { id: 4, x: 700, y: 350, label: "UE-04", phase: Math.PI * 1.5 },
    ];

    const drawAntenna = () => {
      ctx.save();
      ctx.translate(antenna.x, antenna.y);

      // Central Architectural Mast
      ctx.fillStyle = GRAPHITE;
      ctx.fillRect(-1.5, -180, 3, 360);
      
      // Minimalist MIMO Arrays
      for(let i=0; i<4; i++) {
        ctx.fillRect(-15, -140 + i*60, 30, 15);
      }

      // Transmission Pulse
      const t = Date.now() / 1000;
      const s = (Math.sin(t) + 1) / 2;
      ctx.strokeStyle = isPreciseCSI ? NOKIA_BLUE : SIGNAL_RED;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.1 * (1 - s);
      ctx.beginPath();
      ctx.arc(0, -110, 20 + s * 180, 0, Math.PI * 2);
      ctx.stroke();

      ctx.restore();
    };

    const drawDevice = (dev: Device) => {
      ctx.save();
      ctx.translate(dev.x, dev.y);

      // Sleek Smartphone Silhouette
      ctx.strokeStyle = GRAPHITE;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(-12, -22, 24, 44);
      
      // Technical Indicator
      ctx.fillStyle = GRAPHITE;
      ctx.globalAlpha = 0.1;
      ctx.fillRect(-10, -20, 20, 40);
      
      // Active Status
      ctx.globalAlpha = 0.8;
      ctx.fillStyle = isPreciseCSI ? NOKIA_BLUE : SIGNAL_RED;
      ctx.beginPath();
      ctx.arc(0, -15, 2, 0, Math.PI * 2);
      ctx.fill();

      // Label
      ctx.font = '700 9px monospace';
      ctx.fillStyle = GRAPHITE;
      ctx.globalAlpha = 0.4;
      ctx.fillText(dev.label, -15, 38);

      ctx.restore();
    };

    const draw = () => {
      ctx.fillStyle = PAPER_WHITE;
      ctx.fillRect(0, 0, width, height);

      // Grid Background
      ctx.strokeStyle = '#f2f2f2';
      ctx.lineWidth = 1;
      for(let i=0; i<width; i+=120) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, height); ctx.stroke(); }
      for(let i=0; i<height; i+=120) { ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(width, i); ctx.stroke(); }

      // Beamforming Logic
      devices.forEach(dev => {
        const time = Date.now() / 500;
        const color = isPreciseCSI ? NOKIA_BLUE : SIGNAL_RED;
        
        // Jitter calculation for Bad CSI
        const jitter = isPreciseCSI ? 0 : Math.sin(time * 2 + dev.phase) * 15;
        
        ctx.beginPath();
        if (isPreciseCSI) {
          // Precise Beam: Fluid, Solid
          ctx.setLineDash([80, 20]);
          ctx.lineDashOffset = -(Date.now() / 20);
          ctx.lineWidth = 4;
          ctx.globalAlpha = 0.7;
        } else {
          // Degraded Path: Fragmented, Jittery
          ctx.setLineDash([4, 8]);
          ctx.lineDashOffset = -(Date.now() / 40);
          ctx.lineWidth = 1.5;
          ctx.globalAlpha = 0.4;
        }

        ctx.strokeStyle = color;
        
        const cpX = (antenna.x + dev.x) / 2;
        const cpY = (antenna.y + dev.y) / 2 - 120 + jitter;
        
        ctx.moveTo(antenna.x, antenna.y - 110);
        ctx.quadraticCurveTo(cpX, cpY, dev.x, dev.y);
        ctx.stroke();

        // Target Aura (AI mode)
        if (isPreciseCSI) {
            ctx.save();
            ctx.setLineDash([]);
            ctx.globalAlpha = 0.05;
            ctx.fillStyle = NOKIA_BLUE;
            ctx.beginPath();
            ctx.arc(dev.x, dev.y, 40, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        }
      });

      drawAntenna();
      devices.forEach(drawDevice);

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animationFrameId);
  }, [isPreciseCSI]);

  return (
    <section className="py-24 px-6 max-w-7xl mx-auto overflow-hidden">
      <div className="sketch-card p-12 bg-white/95 backdrop-blur-3xl border-2 border-foreground rounded-[3rem] shadow-[20px_20px_0px_0px_rgba(0,0,0,0.02)]">
        <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-12 mb-16">
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-xl ${isPreciseCSI ? 'bg-nokia-blue/10 text-nokia-blue' : 'bg-red-500/10 text-red-500'}`}>
                <Radio className="h-6 w-6" />
              </div>
              <h3 className="text-4xl font-black tracking-tighter uppercase italic">Phase Precision</h3>
            </div>
            <p className="text-sm text-text-muted max-w-xl leading-relaxed font-medium">
              A minimalist study in spectral alignment. Toggle to see how **CSI Prediction** 
              transforms scattered radio energy into coherent, high-velocity data beams.
            </p>
          </div>
          
          <div className="flex p-1.5 bg-gray-50 border-2 border-foreground rounded-[2rem] shadow-[10px_10px_0px_0px_rgba(0,0,0,1)]">
            <button 
              onClick={() => setIsPreciseCSI(false)}
              className={`px-10 py-4 rounded-[1.5rem] text-xs font-black transition-all uppercase tracking-[0.2em] ${!isPreciseCSI ? 'bg-foreground text-white' : 'hover:bg-gray-200'}`}
            >
              Degraded
            </button>
            <button 
              onClick={() => setIsPreciseCSI(true)}
              className={`px-10 py-4 rounded-[1.5rem] text-xs font-black transition-all uppercase tracking-[0.2em] ${isPreciseCSI ? 'bg-nokia-blue text-white' : 'hover:bg-gray-200'}`}
            >
              Precise
            </button>
          </div>
        </div>

        <div className="relative aspect-video w-full rounded-[4rem] overflow-hidden border-2 border-foreground bg-[#FDFDFD]">
          <canvas ref={canvasRef} className="w-full h-full" />
          
          {/* Minimalist Telemetry Readout */}
          <div className="absolute top-12 right-12 flex gap-16">
            <div className="space-y-1">
              <div className="text-[10px] font-bold text-gray-300 uppercase tracking-widest">Signal Variance</div>
              <div className={`text-4xl font-black font-mono leading-none ${isPreciseCSI ? 'text-nokia-blue' : 'text-red-500'}`}>
                {isPreciseCSI ? '0.003' : '0.124'}<span className="text-xs ml-2">NMSE</span>
              </div>
            </div>
            <div className="space-y-1">
              <div className="text-[10px] font-bold text-gray-300 uppercase tracking-widest">Beam Stability</div>
              <div className="text-4xl font-black font-mono text-foreground">
                {isPreciseCSI ? '99.7' : '14.2'}<span className="text-xs ml-2">%</span>
              </div>
            </div>
          </div>

          {/* Environmental Telemetry */}
          <div className="absolute bottom-12 left-12">
            <div className={`px-8 py-4 rounded-full border-2 border-foreground bg-white font-black text-[11px] uppercase tracking-[0.3em] shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] flex items-center gap-4`}>
              <Waves className={`h-4 w-4 ${isPreciseCSI ? 'text-nokia-blue' : 'text-red-500 animate-pulse'}`} />
              {isPreciseCSI ? 'Coherent Wavefront Synthesis' : 'Incoherent Multi-path Jitter'}
            </div>
          </div>
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-2 gap-12">
          <div className="flex gap-6 items-start">
            <div className={`p-4 rounded-2xl border-2 border-foreground ${!isPreciseCSI ? 'bg-red-500 text-white shadow-[6px_6px_0px_0px_rgba(0,0,0,1)]' : 'bg-white opacity-20'}`}>
              <ShieldAlert className="h-6 w-6" />
            </div>
            <div>
              <h4 className="font-bold text-lg mb-2 uppercase tracking-tight">The Jitter Problem</h4>
              <p className="text-sm text-text-muted leading-relaxed font-medium">
                Without accurate CSI, the base station cannot correctly align phases, resulting in energy leakage and signal fragmentation.
              </p>
            </div>
          </div>
          <div className="flex gap-6 items-start">
            <div className={`p-4 rounded-2xl border-2 border-foreground ${isPreciseCSI ? 'bg-nokia-blue text-white shadow-[6px_6px_0px_0px_rgba(0,0,0,1)]' : 'bg-white opacity-20'}`}>
              <CheckCircle2 className="h-6 w-6" />
            </div>
            <div>
              <h4 className="font-bold text-lg mb-2 uppercase tracking-tight">AI Coherence</h4>
              <p className="text-sm text-text-muted leading-relaxed font-medium">
                Diffusion-based prediction provides a near-perfect map of the environment, enabling surgical beamforming and zero interference.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
