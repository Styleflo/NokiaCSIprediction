'use client';

import { useEffect, useRef } from 'react';

export default function AntennaBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let width = window.innerWidth;
    let height = window.innerHeight;

    const NOKIA_BLUE = '#1241C6';
    const GRAPHITE = '#2C2C2C';
    const SIGNAL_RED = '#D64045';

    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width;
      canvas.height = height;
    };

    window.addEventListener('resize', resize);
    resize();

    interface Antenna {
      x: number;
      y: number;
      height: number;
      pulse: number;
    }

    const antennas: Antenna[] = [];
    const numAntennas = 12;

    for (let i = 0; i < numAntennas; i++) {
      antennas.push({
        x: Math.random() * width,
        y: Math.random() * height,
        height: 20 + Math.random() * 40,
        pulse: Math.random() * Math.PI * 2,
      });
    }

    interface Transmission {
      source: Antenna;
      target: Antenna;
      offset: number;
      speed: number;
      quality: 'good' | 'bad';
      cp1x: number;
      cp1y: number;
    }

    const transmissions: Transmission[] = [];
    const numTransmissions = 20;

    const createTransmission = () => {
      const source = antennas[Math.floor(Math.random() * antennas.length)];
      let target = antennas[Math.floor(Math.random() * antennas.length)];
      while (target === source) {
        target = antennas[Math.floor(Math.random() * antennas.length)];
      }

      // Create a curved path (control point)
      const midX = (source.x + target.x) / 2;
      const midY = (source.y + target.y) / 2;
      const dist = Math.sqrt(Math.pow(target.x - source.x, 2) + Math.pow(target.y - source.y, 2));
      
      return {
        source,
        target,
        offset: 0,
        speed: 0.2 + Math.random() * 0.8,
        quality: Math.random() > 0.7 ? 'bad' : 'good',
        cp1x: midX + (Math.random() - 0.5) * dist * 0.5,
        cp1y: midY + (Math.random() - 0.5) * dist * 0.5,
      };
    };

    for (let i = 0; i < numTransmissions; i++) {
      transmissions.push(createTransmission());
    }

    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      // Draw Transmissions (Curves)
      transmissions.forEach((trans) => {
        ctx.beginPath();
        ctx.setLineDash([5, 15]);
        ctx.lineDashOffset = -trans.offset;
        ctx.strokeStyle = trans.quality === 'good' ? NOKIA_BLUE : SIGNAL_RED;
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = trans.quality === 'good' ? 0.15 : 0.35;

        ctx.moveTo(trans.source.x, trans.source.y);
        ctx.quadraticCurveTo(trans.cp1x, trans.cp1y, trans.target.x, trans.target.y);
        ctx.stroke();

        trans.offset += trans.speed;
        
        // Dynamic instability for bad signals
        if (trans.quality === 'bad') {
          trans.offset += Math.sin(Date.now() / 200) * 0.5;
        }
      });

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return <canvas ref={canvasRef} className="antenna-bg" />;
}
