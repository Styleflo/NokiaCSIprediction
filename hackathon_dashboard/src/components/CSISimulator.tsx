'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Radio, Zap, Activity, ShieldAlert, CheckCircle2 } from 'lucide-react';

export default function CSISimulator() {
  const [useCSIPrediction, setUseCSIPrediction] = useState(false);
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

    // Mixed Distribution of Antennas and Users
    const antennas = [
      { x: 200, y: 150, pulse: 0 },
      { x: 500, y: 450, pulse: 1 },
      { x: 850, y: 150, pulse: 2 },
      { x: 1000, y: 400, pulse: 0.5 },
      { x: 300, y: 500, pulse: 1.5 },
    ];

    const users = [
      { id: 1, x: 400, y: 100, speed: 0.4, offset: 0, antIdx: 0, label: "UE-01" },
      { id: 2, x: 650, y: 250, speed: 0.7, offset: 10, antIdx: 1, label: "UE-02" },
      { id: 3, x: 800, y: 500, speed: 0.3, offset: 5, antIdx: 2, label: "UE-03" },
      { id: 4, x: 100, y: 450, speed: 0.6, offset: 20, antIdx: 4, label: "UE-04" },
      { id: 5, x: 1100, y: 200, speed: 0.5, offset: 15, antIdx: 3, label: "UE-05" },
      { id: 6, x: 600, y: 100, speed: 0.8, offset: 30, antIdx: 0, label: "UE-06" },
      { id: 7, x: 900, y: 550, speed: 0.4, offset: 40, antIdx: 3, label: "UE-07" },
      { id: 8, x: 750, y: 300, speed: 0.6, offset: 12, antIdx: 2, label: "UE-08" },
    ];

    const drawMicroAntenna = (ant: { x: number, y: number, pulse: number }) => {
      ctx.save();
      ctx.translate(ant.x, ant.y);

      // Minimalist Mast
      ctx.strokeStyle = GRAPHITE;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(0, 20);
      ctx.stroke();

      // Small Array Panel
      ctx.fillStyle = GRAPHITE;
      ctx.fillRect(-4, -8, 8, 14);

      // Signal Pulse
      const s = (Math.sin(Date.now() / 400 + ant.pulse) + 1) / 2;
      ctx.strokeStyle = useCSIPrediction ? NOKIA_BLUE : SIGNAL_RED;
      ctx.globalAlpha = (useCSIPrediction ? 0.2 : 0.5) * s;
      ctx.beginPath();
      ctx.arc(0, -5, 5 + s * 25, 0, Math.PI * 2);
      ctx.stroke();
      
      ctx.globalAlpha = 1;
      ctx.restore();
    };

    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      // Grid
      ctx.strokeStyle = '#f8f8f8';
      ctx.lineWidth = 1;
      for(let i=0; i<width; i+=120) {
        ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, height); ctx.stroke();
      }
      for(let i=0; i<height; i+=120) {
        ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(width, i); ctx.stroke();
      }

      // Communications
      users.forEach(user => {
        const isBad = !useCSIPrediction;
        const color = isBad ? SIGNAL_RED : NOKIA_BLUE;
        const ant = antennas[user.antIdx];
        
        // Data Path (Downlink)
        ctx.beginPath();
        ctx.setLineDash(isBad ? [2, 6] : [20, 15]);
        ctx.lineDashOffset = -user.offset;
        ctx.strokeStyle = color;
        ctx.lineWidth = isBad ? 2 : 2.5;
        ctx.globalAlpha = isBad ? 0.8 : 0.5;
        
        const cpX = (ant.x + user.x) / 2;
        const cpY = (ant.y + user.y) / 2 - 50;
        
        ctx.moveTo(ant.x, ant.y - 5);
        ctx.quadraticCurveTo(cpX, cpY, user.x, user.y);
        ctx.stroke();

        // Feedback Path (Uplink Overhead)
        if (isBad) {
            ctx.beginPath();
            ctx.setLineDash([4, 4]);
            ctx.lineDashOffset = user.offset * 3;
            ctx.strokeStyle = color;
            ctx.lineWidth = 10; // Extra thick for problem visibility
            ctx.globalAlpha = 0.2;
            ctx.moveTo(user.x, user.y);
            ctx.quadraticCurveTo(cpX, cpY + 80, ant.x, ant.y);
            ctx.stroke();

            // Clogged particles
            for(let i=0; i<2; i++) {
                const t = ((user.offset * 0.04 + i*0.5) % 1);
                const px = user.x * (1-t) + ant.x * t;
                const py = user.y * (1-t) + ant.y * t + Math.sin(t * Math.PI) * 60;
                ctx.fillStyle = SIGNAL_RED;
                ctx.globalAlpha = 0.7;
                ctx.beginPath();
                ctx.arc(px, py, 4, 0, Math.PI * 2);
                ctx.fill();
            }
        } else {
            ctx.beginPath();
            ctx.setLineDash([1, 20]);
            ctx.lineDashOffset = user.offset;
            ctx.strokeStyle = NOKIA_BLUE;
            ctx.lineWidth = 1;
            ctx.globalAlpha = 0.4;
            ctx.moveTo(user.x, user.y);
            ctx.lineTo(ant.x, ant.y - 5);
            ctx.stroke();
        }

        // Device
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = GRAPHITE;
        ctx.lineWidth = 2;
        ctx.strokeRect(user.x - 7, user.y - 11, 14, 22);
        ctx.fillStyle = GRAPHITE;
        ctx.fillRect(user.x - 2, user.y + 6, 4, 1.5);
        
        user.offset += user.speed * (isBad ? 2.8 : 1.3);
      });

      // Antennas
      antennas.forEach(drawMicroAntenna);

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animationFrameId);
  }, [useCSIPrediction]);

  return (
    <div className="sketch-card p-8 bg-white/80 backdrop-blur-sm overflow-hidden">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-8">
        <div>
          <h3 className="text-2xl font-bold flex items-center gap-2 mb-2">
            <Radio className="h-6 w-6 text-nokia-blue" />
            Live Network Simulation
          </h3>
          <p className="text-sm text-text-muted max-w-md">
            Visualizing the trade-off between CSI feedback overhead and reconstruction accuracy.
          </p>
        </div>
        
        <div className="flex gap-2 p-1 bg-background border-2 border-foreground rounded-xl">
          <button 
            onClick={() => setUseCSIPrediction(false)}
            className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${!useCSIPrediction ? 'bg-foreground text-white' : 'hover:bg-gray-100'}`}
          >
            Legacy (FDD)
          </button>
          <button 
            onClick={() => setUseCSIPrediction(true)}
            className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${useCSIPrediction ? 'bg-nokia-blue text-white border-nokia-blue' : 'hover:bg-gray-100'}`}
          >
            With CSI Prediction
          </button>
        </div>
      </div>

      <div className="relative aspect-video w-full rounded-2xl overflow-hidden border-2 border-foreground bg-white mb-8 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
        <canvas ref={canvasRef} className="w-full h-full" />
        
        {/* Real-time Stats Overlay */}
        <div className="absolute top-4 right-4 space-y-2">
          <div className={`p-3 rounded-lg border-2 border-foreground bg-white flex items-center gap-3 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]`}>
            <Activity className={`h-4 w-4 ${useCSIPrediction ? 'text-nokia-blue' : 'text-red-500 animate-pulse'}`} />
            <div className="text-[10px] font-bold uppercase tracking-widest">
              Feedback Load: <span className="text-sm block">{useCSIPrediction ? '0.78 Mbps' : '102.4 Mbps'}</span>
            </div>
          </div>
          <div className={`p-3 rounded-lg border-2 border-foreground bg-white flex items-center gap-3 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]`}>
            <Zap className={`h-4 w-4 ${useCSIPrediction ? 'text-nokia-blue' : 'text-yellow-500'}`} />
            <div className="text-[10px] font-bold uppercase tracking-widest">
              Spectral Efficiency: <span className="text-sm block">{useCSIPrediction ? '24.2 bps/Hz' : '8.4 bps/Hz'}</span>
            </div>
          </div>
        </div>

        {/* Status Badge */}
        <div className="absolute bottom-4 left-4">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border-2 border-foreground bg-white font-bold text-[10px] uppercase shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]`}>
            {useCSIPrediction ? (
              <>
                <CheckCircle2 className="h-3 w-3 text-nokia-blue" />
                System Optimized
              </>
            ) : (
              <>
                <ShieldAlert className="h-3 w-3 text-red-500" />
                Bandwidth Constrained
              </>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-sm leading-relaxed">
        <div className="space-y-4">
          <h4 className="font-bold flex items-center gap-2 underline decoration-nokia-blue decoration-2 underline-offset-4">
            Legacy Challenge
          </h4>
          <p className="text-text-muted">
            In traditional systems, the User Equipment (UE) sends raw CSI matrices back to the base station. 
            This consumes massive uplink bandwidth, creating a "feedback bottleneck" that limits the number of active users.
          </p>
        </div>
        <div className="space-y-4">
          <h4 className="font-bold flex items-center gap-2 underline decoration-nokia-blue decoration-2 underline-offset-4">
            AI-Driven Future
          </h4>
          <p className="text-text-muted">
            Our **Diffusion-based prediction** allows the system to reconstruct high-fidelity CSI using only a tiny fraction of the data. 
            This releases 128x more bandwidth for actual user content.
          </p>
        </div>
      </div>
    </div>
  );
}


    draw();
    return () => cancelAnimationFrame(animationFrameId);
  }, [useCSIPrediction]);

  return (
    <div className="sketch-card p-8 bg-white/80 backdrop-blur-sm overflow-hidden">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-8">
        <div>
          <h3 className="text-2xl font-bold flex items-center gap-2 mb-2">
            <Radio className="h-6 w-6 text-nokia-blue" />
            Live Network Simulation
          </h3>
          <p className="text-sm text-text-muted max-w-md">
            Visualizing the trade-off between CSI feedback overhead and reconstruction accuracy.
          </p>
        </div>
        
        <div className="flex gap-2 p-1 bg-background border-2 border-foreground rounded-xl">
          <button 
            onClick={() => setUseCSIPrediction(false)}
            className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${!useCSIPrediction ? 'bg-foreground text-white' : 'hover:bg-gray-100'}`}
          >
            Legacy (FDD)
          </button>
          <button 
            onClick={() => setUseCSIPrediction(true)}
            className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${useCSIPrediction ? 'bg-nokia-blue text-white border-nokia-blue' : 'hover:bg-gray-100'}`}
          >
            With CSI Prediction
          </button>
        </div>
      </div>

      <div className="relative aspect-video w-full rounded-2xl overflow-hidden border-2 border-foreground bg-white mb-8 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
        <canvas ref={canvasRef} className="w-full h-full" />
        
        {/* Real-time Stats Overlay */}
        <div className="absolute top-4 right-4 space-y-2">
          <div className={`p-3 rounded-lg border-2 border-foreground bg-white flex items-center gap-3 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]`}>
            <Activity className={`h-4 w-4 ${useCSIPrediction ? 'text-nokia-blue' : 'text-red-500 animate-pulse'}`} />
            <div className="text-[10px] font-bold uppercase tracking-widest">
              Feedback Load: <span className="text-sm block">{useCSIPrediction ? '0.78 Mbps' : '102.4 Mbps'}</span>
            </div>
          </div>
          <div className={`p-3 rounded-lg border-2 border-foreground bg-white flex items-center gap-3 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]`}>
            <Zap className={`h-4 w-4 ${useCSIPrediction ? 'text-nokia-blue' : 'text-yellow-500'}`} />
            <div className="text-[10px] font-bold uppercase tracking-widest">
              Spectral Efficiency: <span className="text-sm block">{useCSIPrediction ? '24.2 bps/Hz' : '8.4 bps/Hz'}</span>
            </div>
          </div>
        </div>

        {/* Status Badge */}
        <div className="absolute bottom-4 left-4">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border-2 border-foreground bg-white font-bold text-[10px] uppercase shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]`}>
            {useCSIPrediction ? (
              <>
                <CheckCircle2 className="h-3 w-3 text-nokia-blue" />
                System Optimized
              </>
            ) : (
              <>
                <ShieldAlert className="h-3 w-3 text-red-500" />
                Bandwidth Constrained
              </>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-sm leading-relaxed">
        <div className="space-y-4">
          <h4 className="font-bold flex items-center gap-2 underline decoration-nokia-blue decoration-2 underline-offset-4">
            Legacy Challenge
          </h4>
          <p className="text-text-muted">
            In traditional systems, the User Equipment (UE) sends raw CSI matrices back to the base station. 
            This consumes massive uplink bandwidth, creating a "feedback bottleneck" that limits the number of active users.
          </p>
        </div>
        <div className="space-y-4">
          <h4 className="font-bold flex items-center gap-2 underline decoration-nokia-blue decoration-2 underline-offset-4">
            AI-Driven Future
          </h4>
          <p className="text-text-muted">
            Our **Diffusion-based prediction** allows the system to reconstruct high-fidelity CSI using only a tiny fraction of the data. 
            This releases 128x more bandwidth for actual user content.
          </p>
        </div>
      </div>
    </div>
  );
}
