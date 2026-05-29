'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Radio, ShieldAlert, CheckCircle2 } from 'lucide-react';

interface User {
  id: number;
  x: number;
  y: number;
  speed: number;
  label: string;
}

/**
 * CSISimulator: A simplified, high-fidelity network environment simulation.
 * Focuses on Downlink Efficiency (Base Station -> User) to demonstrate CSI impact.
 */
export default function CSISimulator() {
  const [isAIOptimized, setIsAIOptimized] = useState(false);
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

    // 1. Spatially Distributed Entities (No overlapping)
    const antenna = { x: 180, y: height / 2 };
    const users: User[] = [
      { id: 1, x: 800, y: 120, speed: 0.3, label: "0xAlpha" },
      { id: 2, x: 1000, y: 250, speed: 0.5, label: "0xBeta" },
      { id: 3, x: 850, y: 480, speed: 0.2, label: "0xGamma" },
      { id: 4, x: 650, y: 320, speed: 0.4, label: "0xDelta" },
    ];

    const drawAntenna = (x: number, y: number) => {
      ctx.save();
      ctx.translate(x, y);

      // Minimalist Architectural Totem
      ctx.fillStyle = GRAPHITE;
      ctx.fillRect(-2, -150, 4, 300); // Main mast
      
      // Rectangular Arrays
      for(let i=0; i<3; i++) {
        ctx.fillRect(-15, -120 + i*40, 30, 20);
      }

      // Transmission Aura
      const s = (Math.sin(Date.now() / 800) + 1) / 2;
      ctx.strokeStyle = isAIOptimized ? NOKIA_BLUE : SIGNAL_RED;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.15 * (1-s);
      ctx.beginPath();
      ctx.arc(0, -100, 20 + s * 150, 0, Math.PI * 2);
      ctx.stroke();

      ctx.restore();
    };

    const drawUser = (user: User) => {
      ctx.save();
      ctx.translate(user.x, user.y);

      // Minimalist Device Symbol
      ctx.strokeStyle = GRAPHITE;
      ctx.lineWidth = 2;
      ctx.strokeRect(-12, -20, 24, 40);
      
      ctx.fillStyle = GRAPHITE;
      ctx.fillRect(-1, -15, 2, 4); // Top sensor
      
      // Tag
      ctx.font = '700 10px monospace';
      ctx.globalAlpha = 0.3;
      ctx.fillText(user.label, -20, 38);

      ctx.restore();
    };

    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      // Subtle Background Texture
      ctx.strokeStyle = '#f4f4f4';
      ctx.lineWidth = 1;
      for(let i=0; i<width; i+=150) {
        ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, height); ctx.stroke();
      }

      // Downlink-only Communication Logic
      users.forEach(user => {
        const color = isAIOptimized ? NOKIA_BLUE : SIGNAL_RED;
        const offset = (Date.now() * 0.08 * user.speed) % 100;

        ctx.beginPath();
        // Contrast: Legacy is fragmented, AI is solid/fluid
        ctx.setLineDash(isAIOptimized ? [60, 20] : [4, 12]);
        ctx.lineDashOffset = -offset;
        ctx.strokeStyle = color;
        ctx.lineWidth = isAIOptimized ? 3 : 1;
        ctx.globalAlpha = isAIOptimized ? 0.8 : 0.4;
        
        // Pure geometric paths
        const cpX = (antenna.x + user.x) / 2;
        const cpY = (antenna.y + user.y) / 2 - 120;
        
        ctx.moveTo(antenna.x, antenna.y - 100);
        ctx.quadraticCurveTo(cpX, cpY, user.x, user.y);
        ctx.stroke();
      });

      drawAntenna(antenna.x, antenna.y);
      users.forEach(drawUser);

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animationFrameId);
  }, [isAIOptimized]);

  return (
    <div className="sketch-card p-12 bg-white/95 backdrop-blur-2xl overflow-hidden mb-24">
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-12 mb-16">
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <div className={`w-10 h-10 rounded-2xl flex items-center justify-center ${isAIOptimized ? 'bg-nokia-blue/10 text-nokia-blue' : 'bg-red-500/10 text-red-500'}`}>
              <Radio className="h-6 w-6" />
            </div>
            <h3 className="text-4xl font-black tracking-tighter text-foreground uppercase">Downlink Analysis</h3>
          </div>
          <p className="text-sm text-text-muted max-w-xl leading-relaxed font-medium">
            Visualizing data delivery precision. CSI Prediction allows the base station to focus energy with surgical accuracy, 
            eliminating the jitter and fragmentation seen in legacy systems.
          </p>
        </div>
        
        <div className="flex p-1.5 bg-gray-50 border-2 border-foreground rounded-[2rem] shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
          <button 
            onClick={() => setIsAIOptimized(false)}
            className={`px-10 py-4 rounded-[1.5rem] text-xs font-black transition-all uppercase tracking-[0.2em] ${!isAIOptimized ? 'bg-foreground text-white' : 'hover:bg-gray-200'}`}
          >
            Legacy
          </button>
          <button 
            onClick={() => setIsAIOptimized(true)}
            className={`px-10 py-4 rounded-[1.5rem] text-xs font-black transition-all uppercase tracking-[0.2em] ${isAIOptimized ? 'bg-nokia-blue text-white' : 'hover:bg-gray-200'}`}
          >
            AI Active
          </button>
        </div>
      </div>

      <div className="relative aspect-video w-full rounded-[4rem] overflow-hidden border-2 border-foreground bg-white shadow-[30px_30px_0px_0px_rgba(0,0,0,0.03)]">
        <canvas ref={canvasRef} className="w-full h-full" />
        
        {/* Minimalist Data Readout */}
        <div className="absolute top-12 right-12 flex gap-12">
          <div className="space-y-1">
            <div className="text-[10px] font-bold text-gray-300 uppercase tracking-widest">Network Load</div>
            <div className={`text-4xl font-black font-mono ${isAIOptimized ? 'text-nokia-blue' : 'text-red-500'}`}>
              {isAIOptimized ? '0.78' : '102.4'}<span className="text-xs ml-2">MB/S</span>
            </div>
          </div>
          <div className="space-y-1">
            <div className="text-[10px] font-bold text-gray-300 uppercase tracking-widest">Efficiency</div>
            <div className="text-4xl font-black font-mono text-foreground">
              {isAIOptimized ? '+128' : '0'}<span className="text-xs ml-2">X</span>
            </div>
          </div>
        </div>

        {/* Dynamic Status Badge */}
        <div className="absolute bottom-12 left-12">
          <div className={`flex items-center gap-4 px-8 py-4 rounded-full border-2 border-foreground bg-white font-black text-xs uppercase tracking-[0.3em] shadow-[10px_10px_0px_0px_rgba(0,0,0,1)]`}>
            {isAIOptimized ? (
              <>
                <CheckCircle2 className="h-5 w-5 text-nokia-blue" />
                Optimized Beamforming
              </>
            ) : (
              <>
                <ShieldAlert className="h-5 w-5 text-red-500" />
                Signal Fragmentation
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
