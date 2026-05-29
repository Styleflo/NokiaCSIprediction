import Image from "next/image";
import { Cpu, Zap, Target, TrendingDown, BookOpen, Network, Signal, ShieldCheck } from "lucide-react";
import NetworkSimulation from "@/components/NetworkSimulation";

export default function Home() {
  return (
    <div className="relative">
      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6 max-w-6xl mx-auto text-center" id="abstract">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-nokia-blue/5 border border-nokia-blue/10 text-nokia-blue text-[10px] font-bold uppercase tracking-widest mb-8 animate-pulse-soft">
          <Signal className="h-3 w-3" />
          5G Advanced & 6G Research
        </div>
        <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-gray-900 mb-8 leading-tight">
          Redefining <span className="text-nokia-blue">CSI Feedback</span> <br /> 
          with Generative AI
        </h1>
        <p className="max-w-3xl mx-auto text-lg text-text-muted leading-relaxed mb-12">
          Reducing Channel State Information (CSI) overhead is the "holy grail" of Massive MIMO. 
          Our research leverages Diffusion models to compress feedback by 128x while maintaining 
          unprecedented reconstruction accuracy.
        </p>
        <div className="flex justify-center gap-4">
          <a href="#results" className="px-8 py-3 bg-nokia-blue text-white rounded-full font-semibold shadow-lg shadow-nokia-blue/20 hover:bg-nokia-blue/90 transition-all">
            View Research Results
          </a>
          <a href="#methodology" className="px-8 py-3 bg-white border border-accent-muted rounded-full font-semibold hover:border-nokia-blue/30 transition-all">
            Technical Methodology
          </a>
        </div>
      </section>

      {/* Why Section - The Problem Context */}
      <section className="py-24 px-6 bg-white/40 backdrop-blur-sm border-y border-accent-muted">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-16 items-center">
            <div className="space-y-8">
              <h2 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
                <Network className="h-8 w-8 text-nokia-blue" />
                The Efficiency Bottleneck
              </h2>
              <div className="space-y-6 text-text-muted">
                <p>
                  In Massive MIMO, the Base Station needs precise knowledge of the wireless channel (CSI) 
                  to focus energy toward specific users. However, in FDD mode, the UE must "feedback" 
                  this high-dimensional data back to the station.
                </p>
                <div className="p-6 bg-nokia-blue-soft rounded-2xl border border-nokia-blue/5">
                  <span className="block text-2xl font-bold text-nokia-blue mb-1">90%</span>
                  <p className="text-sm">of uplink bandwidth can be consumed just by CSI feedback in dense antenna arrays, leaving almost no room for user data.</p>
                </div>
                <p>
                  Our goal was to find a "latent representation" that captures the essence of the channel 
                  in just a few bits, effectively solving the trade-off between accuracy and overhead.
                </p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
               {[
                 { icon: Signal, title: "Latency", text: "Delayed CSI leads to beam misalignment." },
                 { icon: ShieldCheck, title: "Accuracy", text: "Lossy compression degrades signal SNR." },
                 { icon: Cpu, title: "Compute", text: "On-device AI must be power efficient." },
                 { icon: Target, title: "Spectral", text: "Maximizing bits per second per Hz." }
               ].map((item, idx) => (
                 <div key={idx} className="glass-card p-6 rounded-3xl">
                   <item.icon className="h-6 w-6 text-nokia-blue mb-4" />
                   <h3 className="font-bold text-sm mb-2">{item.title}</h3>
                   <p className="text-xs text-text-muted leading-relaxed">{item.text}</p>
                 </div>
               ))}
            </div>
          </div>
        </div>
      </section>

      {/* Network Simulation Section */}
      <NetworkSimulation />

      {/* Results Section - The Graphics */}
      <section className="py-24 px-6 max-w-7xl mx-auto" id="results">
        <div className="text-center mb-20">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">Empirical Performance</h2>
          <p className="text-text-muted text-lg">Diffusion models consistently outperformed baseline architectures in reconstruction fidelity.</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 mb-20">
          {/* Main Result Card - High Fidelity Matrix */}
          <div className="glass-card rounded-[3rem] p-10 md:p-14 border-2 border-nokia-blue/20 shadow-2xl shadow-nokia-blue/5">
            <div className="mb-12 space-y-4">
              <div className="inline-block px-4 py-1 rounded-full bg-nokia-blue/10 text-nokia-blue text-[12px] font-bold uppercase tracking-widest">
                Highest Fidelity Reconstruction
              </div>
              <h3 className="text-4xl font-bold">Matrix Accuracy</h3>
              <p className="text-lg text-text-muted">
                Visualizing the precision of our Diffusion Model. The reconstructed CSI matrix captures the intricate spatial-temporal patterns with near-zero deviation from the ground truth.
              </p>
            </div>
            <div className="relative aspect-video w-full rounded-3xl overflow-hidden shadow-inner ring-1 ring-gray-200 bg-white">
              <Image 
                src="/images/metrics/output_unet_matrix.png" 
                alt="CSI Matrix Prediction" 
                fill
                className="object-contain p-8"
              />
            </div>
          </div>

          {/* Loss Curve Card */}
          <div className="glass-card rounded-[3rem] p-10 md:p-14 border-2 border-nokia-blue/20 shadow-2xl shadow-nokia-blue/5">
            <div className="mb-12 space-y-4">
              <div className="inline-block px-4 py-1 rounded-full bg-nokia-blue/10 text-nokia-blue text-[12px] font-bold uppercase tracking-widest">
                Training Dynamics
              </div>
              <h3 className="text-4xl font-bold">Convergence Rate</h3>
              <p className="text-lg text-text-muted">
                Monitoring the NMSE reduction over 200 epochs. The Conv3D + CBAM model exhibits rapid and stable convergence, proving its efficiency for real-time training scenarios.
              </p>
            </div>
            <div className="relative aspect-video w-full rounded-3xl overflow-hidden shadow-inner ring-1 ring-gray-200 bg-white">
              <Image 
                src="/images/metrics/output_final_loss_curve.png" 
                alt="Final Loss Curve" 
                fill
                className="object-contain p-8"
              />
            </div>
          </div>
        </div>

        {/* Highlighted Best Models Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-20">
          {[
            { 
              name: "Diffusion (DDPM)", 
              nmse: "0.0031", 
              desc: "Best overall accuracy. Captures complex channel distributions through iterative refinement.",
              highlight: "border-nokia-blue shadow-nokia-blue/10",
              badge: "bg-nokia-blue text-white"
            },
            { 
              name: "Conv3D + CBAM", 
              nmse: "0.0033", 
              desc: "Optimal trade-off between speed and precision. Perfect for high-mobility scenarios.",
              highlight: "border-cyber-cyan/40 shadow-cyber-cyan/5",
              badge: "bg-cyber-cyan text-white"
            }
          ].map((model, idx) => (
            <div key={idx} className={`glass-card p-10 rounded-[2.5rem] border-2 ${model.highlight} shadow-xl relative overflow-hidden`}>
              <div className={`absolute top-0 right-0 px-6 py-2 rounded-bl-3xl text-[10px] font-bold uppercase tracking-widest ${model.badge}`}>
                Top Performer
              </div>
              <h4 className="text-2xl font-bold mb-4">{model.name}</h4>
              <p className="text-text-muted mb-8 leading-relaxed">{model.desc}</p>
              <div className="flex items-end gap-2">
                <span className="text-5xl font-bold text-nokia-blue">{model.nmse}</span>
                <span className="text-xs text-text-muted font-bold uppercase mb-2">NMSE Score</span>
              </div>
            </div>
          ))}
        </div>

        {/* Leaderboard */}
        <div className="glass-card rounded-[3rem] overflow-hidden border border-accent-muted shadow-sm">
          <div className="px-8 py-6 border-b border-accent-muted flex items-center justify-between">
            <h3 className="font-bold flex items-center gap-2">
              <Target className="h-5 w-5 text-nokia-blue" />
              Benchmark Analysis
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="bg-nokia-blue-soft/50">
                  <th className="px-8 py-4 font-bold text-gray-500 uppercase text-[10px] tracking-widest">Architecture</th>
                  <th className="px-8 py-4 font-bold text-gray-500 uppercase text-[10px] tracking-widest">NMSE (↓)</th>
                  <th className="px-8 py-4 font-bold text-gray-500 uppercase text-[10px] tracking-widest">Latent Dim</th>
                  <th className="px-8 py-4 font-bold text-gray-500 uppercase text-[10px] tracking-widest">Gain vs Baseline</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-accent-muted">
                {[
                  { name: "Diffusion (DDPM)", nmse: "0.0031", dim: "32", gain: "+88.9%" },
                  { name: "Conv3D + CBAM", nmse: "0.0033", dim: "32", gain: "+88.2%" },
                  { name: "CNN + Transformer", nmse: "0.0045", dim: "64", gain: "+83.9%" },
                  { name: "ConvLSTM", nmse: "0.0120", dim: "256", gain: "+57.1%" },
                  { name: "CSI futur = dernier CSI connu", nmse: "0.0280", dim: "-", gain: "0.0%" },
                ].map((row, idx) => (
                  <tr key={idx} className={idx === 0 ? "bg-nokia-blue/5" : "hover:bg-gray-50 transition-colors"}>
                    <td className="px-8 py-5 font-bold">{row.name}</td>
                    <td className="px-8 py-5 font-mono text-nokia-blue">{row.nmse}</td>
                    <td className="px-8 py-5 font-mono">{row.dim}</td>
                    <td className="px-8 py-5 font-bold text-gray-400">{row.gain}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Methodology Section */}
      <section className="py-24 px-6 max-w-6xl mx-auto" id="methodology">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-16">
          <div className="space-y-6">
            <h2 className="text-3xl font-bold">Research <span className="text-nokia-blue">Methodology</span></h2>
            <p className="text-text-muted leading-relaxed">
              Our approach follows a structured pipeline from raw CSI acquisition to real-time reconstruction. 
              We utilize a custom-weighted NMSE loss function to prioritize high-energy subcarriers.
            </p>
            <div className="space-y-4">
               {[
                 { title: "Data Preparation", text: "Complex CSI matrices converted to polar coordinates for better neural alignment." },
                 { title: "Encoder Optimization", text: "Feature projection into a highly compressed 32-dimensional latent vector." },
                 { title: "Diffusion Decoder", text: "Reverse-diffusion process to reconstruct the full matrix from the latent seed." }
               ].map((step, idx) => (
                 <div key={idx} className="flex gap-4 p-4 rounded-2xl bg-white border border-accent-muted shadow-sm">
                   <div className="h-8 w-8 rounded-full bg-nokia-blue text-white flex items-center justify-center font-bold text-xs flex-none">{idx+1}</div>
                   <div>
                     <h4 className="font-bold text-sm mb-1">{step.title}</h4>
                     <p className="text-xs text-text-muted">{step.text}</p>
                   </div>
                 </div>
               ))}
            </div>
          </div>
          <div className="glass-card rounded-[2.5rem] p-12 border-nokia-blue/10 flex flex-col items-center justify-center text-center">
             <BookOpen className="h-12 w-12 text-nokia-blue mb-6" />
             <h3 className="text-2xl font-bold mb-4">Read the Publication</h3>
             <p className="text-sm text-text-muted mb-8">
               Detailed mathematical breakdown of the Diffusion denoising process and attention-mechanism integration.
             </p>
             <button className="px-6 py-2 border border-nokia-blue text-nokia-blue rounded-full font-bold text-sm hover:bg-nokia-blue hover:text-white transition-all cursor-not-allowed opacity-50">
               PDF Coming Soon
             </button>
          </div>
        </div>
      </section>
    </div>
  );
}
