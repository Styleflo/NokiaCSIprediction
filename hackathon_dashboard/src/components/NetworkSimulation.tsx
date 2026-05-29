'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Radio, Zap, Activity, ShieldAlert, CheckCircle2, Info, Maximize2 } from 'lucide-react';

interface User {
  id: number;
  x: number;
  y: number;
  speed: number;
  label: string;
  color: string;
}

/**
 * NetworkSimulation: A dedicated section showcasing the CSI Prediction use case.
 * Designed with the 'environment-simulator' skill for maximum visual impact.
 */
export default function NetworkSimulation() {
  const [isAIPredictionActive, setIsAIPredictionActive] = useState(false);
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

    // Simulation Entities
    const antenna = { x: 200, y: height / 2 };
    const users: User[] = [
      { id: 1, x: 850, y: 150, speed: 0.4, label: "0x00A1", color: '#FF9F1C' },
      { id: 2, x: 1000, y: 300, speed: 0.6, label: "0x00B2", color: '#2EC4B6' },
      { id: 3, x: 900, y: 480, speed: 0.3, label: "0x00C3", color: '#E71D36' },
      { id: 4, x: 700, y: 400, speed: 0.5, label: "0x00D4", color: '#011627' },
    ];

    const drawAntenna = () => {
      ctx.save();
      ctx.translate(antenna.x, antenna.y);

      // Base Station Totem
      ctx.fillStyle = GRAPHITE;
      ctx.fillRect(-4, -180, 8, 360);
      
      // Rectangular Sectorized Arrays
      for(let i=0; i<4; i++) {
        ctx.fillRect(-20, -140 + i*50, 40, 30);
        ctx.strokeStyle = 'rgba(255,255,255,0.2)';
        ctx.strokeRect(-18, -138 + i*50, 36, 26);
      }

      // Transmission Pulse
      const s = (Math.sin(Date.now() / 700) + 1) / 2;
      ctx.strokeStyle = isAIPredictionActive ? NOKIA_BLUE : SIGNAL_RED;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.2 * (1 - s);
      ctx.beginPath();
      ctx.arc(0, -100, 30 + s * 200, 0, Math.PI * 2);
      ctx.stroke();

      ctx.restore();
    };

    const drawUser = (user: User) => {
      ctx.save();
      ctx.translate(user.x, user.y);

      // Sleek Modern Device
      ctx.strokeStyle = GRAPHITE;
      ctx.lineWidth = 2;
      ctx.strokeRect(-15, -25, 30, 50);
      
      // Screen & Dynamic Activity
      ctx.fillStyle = GRAPHITE;
      ctx.fillRect(-12, -20, 24, 40);
      
      const activity = (Math.sin(Date.now() / 200 + user.id) + 1) / 2;
      ctx.fillStyle = isAIPredictionActive ? NOKIA_BLUE : SIGNAL_RED;
      ctx.globalAlpha = 0.4 + activity * 0.6;
      ctx.fillRect(-10, -18, 20, 2);

      // Technical Label
      ctx.font = 'bold 10px monospace';
      ctx.fillStyle = GRAPHITE;
      ctx.globalAlpha = 0.4;
      ctx.fillText(user.label, -18, 42);

      ctx.restore();
    };

    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      // Geometric Background
      ctx.strokeStyle = '#f8f8f8';
      ctx.lineWidth = 1;
      for(let i=0; i<width; i+=100) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, height); ctx.stroke(); }
      for(let i=0; i<height; i+=100) { ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(width, i); ctx.stroke(); }

      // Data Flow Logic
      users.forEach(user => {
        const offset = (Date.now() * 0.1 * user.speed) % 100;
        const color = isAIPredictionActive ? NOKIA_BLUE : SIGNAL_RED;

        // Downlink Path (Architectural Curve)
        ctx.beginPath();
        ctx.setLineDash(isAIPredictionActive ? [40, 15] : [4, 10]);
        ctx.lineDashOffset = -offset;
        ctx.strokeStyle = color;
        ctx.lineWidth = isAIPredictionActive ? 4 : 1.5;
        ctx.globalAlpha = isAIPredictionActive ? 0.9 : 0.5;

        const cpX = (antenna.x + user.x) / 2;
        const cpY = (antenna.y + user.y) / 2 - 150;

        ctx.moveTo(antenna.x, antenna.y - 100);
        ctx.quadraticCurveTo(cpX, cpY, user.x, user.y);
        ctx.stroke();

        // Bandwidth Saturation Effect (Legacy only)
        if (!isAIPredictionActive) {
            ctx.beginPath();
            ctx.setLineDash([1, 1]);
            ctx.lineDashOffset = offset * 3;
            ctx.strokeStyle = SIGNAL_RED;
            ctx.lineWidth = 15;
            ctx.globalAlpha = 0.05;
            ctx.moveTo(user.x, user.y);
            ctx.quadraticCurveTo(cpX, cpY + 250, antenna.x, antenna.y);
            ctx.stroke();
        }
      });

      drawAntenna();
      users.forEach(drawUser);

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animationFrameId);
  }, [isAIPredictionActive]);

  return (
    <section className="py-24 px-6 max-w-7xl mx-auto overflow-hidden">
      <div className="sketch-card p-12 bg-white/95 backdrop-blur-2xl">
        <div className="flex flex-col xl:flex-row justify-between items-start xl:items-center gap-10 mb-16">
          <div className="space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-nokia-blue/10 text-nokia-blue text-[10px] font-bold uppercase tracking-[0.2em]">
              <Activity className="h-3 w-3" /> Live Use-Case Simulation
            </div>
            <h2 className="text-5xl font-black tracking-tighter uppercase leading-none">
              The <span className="text-nokia-blue">CSI Prediction</span> Advantage
            </h2>
            <p className="text-sm text-text-muted max-w-2xl leading-relaxed font-medium">
              Interact with our massive MIMO environment. Switch to **AI Active** to witness the transition from bandwidth saturation 
              to near-instantaneous latent synthesis.
            </p>
          </div>
          
          <div className="flex flex-col gap-6 w-full xl:w-auto">
            <div className="flex p-2 bg-gray-50 border-2 border-foreground rounded-[2rem] shadow-[10px_10px_0px_0px_rgba(0,0,0,1)] self-start">
              <button 
                onClick={() => setIsAIPredictionActive(false)}
                className={`px-12 py-5 rounded-[1.5rem] text-xs font-black transition-all uppercase tracking-[0.25em] ${!isAIPredictionActive ? 'bg-foreground text-white' : 'hover:bg-gray-200'}`}
              >
                Legacy FDD
              </button>
              <button 
                onClick={() => setIsAIPredictionActive(true)}
                className={`px-12 py-5 rounded-[1.5rem] text-xs font-black transition-all uppercase tracking-[0.25em] ${isAIPredictionActive ? 'bg-nokia-blue text-white shadow-lg shadow-nokia-blue/20' : 'hover:bg-gray-200'}`}
              >
                AI Optimized
              </button>
            </div>
          </div>
        </div>

        <div className="relative aspect-[21/9] w-full rounded-[4.5rem] overflow-hidden border-2 border-foreground bg-white shadow-[40px_40px_0px_0px_rgba(0,0,0,0.02)]">
          <canvas ref={canvasRef} className="w-full h-full" />
          
          {/* Real-time Dashboard Readout */}
          <div className="absolute top-12 right-12 flex gap-12 bg-white/80 backdrop-blur-md p-8 rounded-[2.5rem] border-2 border-foreground shadow-[10px_10px_0px_0px_rgba(0,0,0,1)]">
            <div className="space-y-1">
              <div className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Feedback Overhead</div>
              <div className={`text-4xl font-black font-mono leading-none ${isAIPredictionActive ? 'text-nokia-blue' : 'text-red-500'}`}>
                {isAIPredictionActive ? '0.78' : '102.4'}<span className="text-xs ml-2">MB/S</span>
              </div>
            </div>
            <div className="w-px h-12 bg-gray-200" />
            <div className="space-y-1">
              <div className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Spectral Boost</div>
              <div className="text-4xl font-black font-mono leading-none text-foreground">
                {isAIPredictionActive ? '128' : '1'}<span className="text-xs ml-2">X</span>
              </div>
            </div>
          </div>

          {/* Environmental Telemetry */}
          <div className="absolute bottom-12 left-12 flex items-center gap-6">
            <div className={`px-8 py-4 rounded-full border-2 border-foreground bg-white font-black text-[11px] uppercase tracking-[0.3em] shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] flex items-center gap-3`}>
              <div className={`w-2 h-2 rounded-full ${isAIPredictionActive ? 'bg-nokia-blue' : 'bg-red-500 animate-pulse'}`} />
              {isAIPredictionActive ? 'AI Inference: Active' : 'Uplink Congestion: High'}
            </div>
            <div className="flex gap-2">
              {[1, 2, 3, 4].map(i => (
                <div key={i} className={`w-1.5 h-1.5 rounded-full border border-foreground ${isAIPredictionActive ? 'bg-nokia-blue' : 'bg-transparent'}`} />
              ))}
            </div>
          </div>

          <button className="absolute bottom-12 right-12 p-4 rounded-2xl border-2 border-foreground bg-white hover:bg-gray-50 transition-colors shadow-[6px_6px_0px_0px_rgba(0,0,0,1)]">
            <Maximize2 className="h-5 w-5" />
          </button>
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-2 gap-12">
          <div className="p-8 rounded-[2.5rem] bg-gray-50 border-2 border-accent-muted flex gap-6 items-start">
            <div className="p-4 rounded-2xl bg-white border border-accent-muted shadow-sm">
              <ShieldAlert className="h-6 w-6 text-red-500" />
            </div>
            <div>
              <h4 className="font-bold text-lg mb-2 uppercase tracking-tight">The Legacy Clog</h4>
              <p className="text-sm text-text-muted leading-relaxed font-medium">
                In traditional FDD systems, the User Equipment (UE) reports raw channel matrices back to the station. 
                With 64+ antennas, this "feedback" tax consumes up to 90% of available bandwidth.
              </p>
            </div>
          </div>
          <div className="p-8 rounded-[2.5rem] bg-nokia-blue/5 border-2 border-nokia-blue/20 flex gap-6 items-start text-nokia-blue">
            <div className="p-4 rounded-2xl bg-white border border-nokia-blue/10 shadow-sm">
              <CheckCircle2 className="h-6 w-6" />
            </div>
            <div>
              <h4 className="font-bold text-lg mb-2 uppercase tracking-tight">AI Synthesis</h4>
              <p className="text-sm text-nokia-blue/70 leading-relaxed font-medium">
                By sending only a 32-dimensional latent seed, the Base Station uses a Diffusion-based generative model 
                to "hallucinate" the high-fidelity CSI, recovering nearly all wasted bandwidth.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
