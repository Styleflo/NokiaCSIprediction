'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Radio, Zap, ShieldAlert, CheckCircle2, Waves, Activity } from 'lucide-react';

interface Device {
  id: number;
  x: number;
  y: number;
  label: string;
  offset: number;
}

/**
 * NetworkSimulation: Recreated based on the reference image.
 * Demonstrates the transition from Poor/Delayed CSI to Accurate/Real-time CSI.
 */
export default function NetworkSimulation() {
  const [isCSIPredictionEnabled, setIsCSIPredictionActive] = useState(false);
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

    const antenna = { x: 150, y: height / 2 };
    const devices: Device[] = [
      { id: 1, x: 900, y: 120, label: "DEVICE M1", offset: 0 },
      { id: 2, x: 950, y: 300, label: "DEVICE M2", offset: 10 },
      { id: 3, x: 900, y: 480, label: "DEVICE M3", offset: 20 },
    ];

    const drawAntenna = () => {
      ctx.save();
      ctx.translate(antenna.x, antenna.y);

      // Mast (Tripod style from image)
      ctx.strokeStyle = GRAPHITE;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, 0); ctx.lineTo(0, -60); // Vertical
      ctx.moveTo(0, 0); ctx.lineTo(-20, 40); // Leg 1
      ctx.moveTo(0, 0); ctx.lineTo(20, 40); // Leg 2
      ctx.stroke();

      // Top detail
      ctx.fillStyle = GRAPHITE;
      ctx.beginPath();
      ctx.arc(0, -60, 5, 0, Math.PI * 2);
      ctx.fill();

      // Radio waves icon detail
      ctx.beginPath();
      ctx.arc(0, -60, 12, -Math.PI/1.5, -Math.PI/3); ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, -60, 18, -Math.PI/1.5, -Math.PI/3); ctx.stroke();

      ctx.restore();
    };

    const drawDevice = (dev: Device) => {
      ctx.save();
      ctx.translate(dev.x, dev.y);

      // Device body
      ctx.strokeStyle = GRAPHITE;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(-12, -22, 24, 44);
      
      // Screen area
      ctx.fillStyle = GRAPHITE;
      ctx.globalAlpha = 0.05;
      ctx.fillRect(-10, -20, 20, 40);
      
      // Label
      ctx.font = '800 10px monospace';
      ctx.fillStyle = GRAPHITE;
      ctx.globalAlpha = 0.6;
      ctx.fillText(dev.label, -25, 40);

      // Noise/Particles for bad CSI
      if (!isCSIPredictionEnabled) {
          ctx.globalAlpha = 0.4;
          for(let i=0; i<8; i++) {
              const r = 20 + Math.sin(Date.now()/200 + i) * 5;
              const angle = i * Math.PI / 4;
              ctx.beginPath();
              ctx.arc(Math.cos(angle)*r, Math.sin(angle)*r, 1, 0, Math.PI*2);
              ctx.fill();
          }
      }

      ctx.restore();
    };

    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      // Lined Paper Effect (from image)
      ctx.strokeStyle = '#e5e5e5';
      ctx.lineWidth = 1;
      for(let i=0; i<height; i+=30) {
        ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(width, i); ctx.stroke();
      }

      // Vertical Divider
      ctx.strokeStyle = GRAPHITE;
      ctx.globalAlpha = 0.1;
      ctx.beginPath(); ctx.moveTo(width/2, 50); ctx.lineTo(width/2, height-50); ctx.stroke();
      ctx.globalAlpha = 1;

      devices.forEach(dev => {
        const time = Date.now() / 1000;
        const color = isCSIPredictionEnabled ? NOKIA_BLUE : GRAPHITE;
        const jitter = isCSIPredictionEnabled ? 0 : Math.sin(time * 10 + dev.id) * 8;
        
        ctx.beginPath();
        if (isCSIPredictionEnabled) {
          // Precise Path
          ctx.setLineDash([]);
          ctx.lineWidth = 2.5;
          ctx.globalAlpha = 0.7;
          ctx.strokeStyle = NOKIA_BLUE;
        } else {
          // Poor Path
          ctx.setLineDash([4, 6]);
          ctx.lineWidth = 1;
          ctx.globalAlpha = 0.3;
          ctx.strokeStyle = GRAPHITE;
        }

        const cpX = (antenna.x + dev.x) / 2;
        const cpY = (antenna.y + dev.y) / 2 - 80 + jitter;
        
        ctx.moveTo(antenna.x, antenna.y - 60);
        ctx.quadraticCurveTo(cpX, cpY, dev.x, dev.y);
        ctx.stroke();

        // Annotation Text (from image)
        ctx.save();
        ctx.font = 'italic 10px monospace';
        ctx.fillStyle = GRAPHITE;
        ctx.globalAlpha = 0.5;
        const textX = (antenna.x + dev.x) / 2;
        const textY = (antenna.y + dev.y) / 2 - 60 + jitter;
        const label = isCSIPredictionEnabled ? "ERROR: MINIMAL / DELAY: 2ms" : "ERROR: HIGH / DELAY: 20ms";
        ctx.fillText(label, textX - 40, textY);
        ctx.restore();
      });

      drawAntenna();
      devices.forEach(drawDevice);

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animationFrameId);
  }, [isCSIPredictionEnabled]);

  return (
    <section className="py-24 px-6 max-w-7xl mx-auto overflow-hidden">
      <div className="sketch-card p-12 bg-white border-2 border-foreground relative shadow-[16px_16px_0px_0px_rgba(0,0,0,1)]">
        
        {/* Title Header */}
        <div className="flex justify-between items-center mb-12 border-b-2 border-foreground pb-8">
          <div>
            <h3 className="text-3xl font-black tracking-tight uppercase">Network Simulation: The Effect of CSI Prediction</h3>
            <p className="text-[11px] font-bold text-gray-400 mt-1 uppercase tracking-widest">
              MODE: {isCSIPredictionEnabled ? 'ACCURATE / REAL-TIME CSI' : 'POOR / DELAYED CSI'}
            </p>
          </div>
          
          <button 
            onClick={() => setIsCSIPredictionActive(!isCSIPredictionEnabled)}
            className={`px-8 py-4 rounded-full border-2 border-foreground font-black text-xs uppercase tracking-widest transition-all shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] active:translate-x-[2px] active:translate-y-[2px] active:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] ${isCSIPredictionEnabled ? 'bg-nokia-blue text-white' : 'bg-white text-foreground'}`}
          >
            Switch Prediction Mode
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-12">
          {/* Main Simulation Area */}
          <div className="lg:col-span-3 relative aspect-video w-full rounded-2xl overflow-hidden border-2 border-foreground bg-[#FDFDFD]">
            <canvas ref={canvasRef} className="w-full h-full" />
            
            {/* Legend Overlays */}
            <div className="absolute bottom-8 left-8 space-y-2">
                <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest">
                    <span className="w-4 h-[2px] bg-foreground"></span> THROUGHPUT: {isCSIPredictionEnabled ? 'HIGH' : 'LOW'}
                </div>
                <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest">
                    <span className="w-4 h-[2px] bg-foreground"></span> CONNECTION: {isCSIPredictionEnabled ? 'OPTIMAL' : 'POOR'}
                </div>
                <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest">
                    <span className="w-4 h-[2px] bg-foreground"></span> SIGNAL POWER: {isCSIPredictionEnabled ? 'MAXIMIZED' : 'FADED'}
                </div>
            </div>
          </div>

          {/* Metrics Panel (from image) */}
          <div className="space-y-6">
            <div className="p-6 border-2 border-foreground bg-white shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
              <div className="flex items-center gap-2 mb-4 border-b border-gray-100 pb-2">
                <Activity className="h-4 w-4 text-nokia-blue" />
                <span className="text-[10px] font-bold uppercase tracking-widest">Metric 1: Throughput</span>
              </div>
              <div className="text-4xl font-black font-mono">
                {isCSIPredictionEnabled ? '580' : '42'}<span className="text-xs ml-2">Mbps (avg)</span>
              </div>
            </div>

            <div className="p-6 border-2 border-foreground bg-white shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
              <div className="flex items-center gap-2 mb-4 border-b border-gray-100 pb-2">
                <Zap className="h-4 w-4 text-yellow-500" />
                <span className="text-[10px] font-bold uppercase tracking-widest">Metric 2: Quality</span>
              </div>
              <div className="text-4xl font-black font-mono">
                {isCSIPredictionEnabled ? '98' : '12'}<span className="text-xs ml-2">% STABLE</span>
              </div>
            </div>

            <div className="p-6 rounded-2xl bg-gray-50 border border-dashed border-gray-300">
                <p className="text-[10px] font-medium leading-relaxed text-gray-500 italic">
                    Note: Simulation reflects empirical performance of Diffusion-based prediction in high-mobility scenarios.
                </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
