# Implementation Plan: Nokia Hackathon Dashboard (Next.js & Vercel Optimized)

## Objective
Build a high-performance, production-ready **Next.js (App Router)** web application styled with **Tailwind CSS** to showcase the Nokia Hackathon results. The dashboard will feature an immersive, premium Nokia-branded UI/UX to highlight the cutting-edge AI models developed for Channel State Information (CSI) prediction in Massive MIMO systems.

---

## Brand Identity & UI Theme (Nokia Inspired)
To reflect Nokia's corporate identity and the high-tech spirit of the hackathon, the application will follow these design guidelines:
* **Colors:** * Primary: Nokia Pure Blue (`#005AFF` or `#1241C6`)
    * Backgrounds: Clean White (`#FFFFFF`) and Tech Slate Dark/Light modes.
    * Accents: Cyber Cyan and Deep Navy for data visualizations.
* **Assets:** Integration of the official **Nokia Logo** in the navigation bar and a dedicated "Hackathon Edition" badge.
* **Feel:** Minimalist, fast, professional, and data-driven.

---

## Key Files & Context
* **Target Directory:** `hackathon_dashboard` (Next.js project root)
* **Source Material:** Insights gathered from `Rendu/rendu.tex`, `model_training/*.ipynb`
* **Assets to Migrate:** * `Rendu/output_final_loss_curve.png`
    * `Rendu/output_unet_matrix.png`
    * *New:* Nokia Logo SVG / Hackathon Banner asset.

---

## Implementation Steps

### 1. Initialize Project (Next.js + TypeScript)
* Scaffold the project using the official Next.js template optimized for Vercel:
    ```bash
    npx create-next-app@latest nokia-hackathon-dashboard --ts --tailwind --eslint --app --src-dir
    ```
* Ensure the App Router structure is selected for optimized layouts and Server Components.

### 2. Setup Dependencies & Vercel Optimization
* Install Lucide React for modern, clean tech icons: `npm install lucide-react`
* Configure `next.config.js` for image optimization and Vercel analytics readiness.
* *(Optional)* Install Shadcn/ui or Radix primitives for high-quality dashboard components (tables, tabs).

### 3. Asset Migration & Public Folder Setup
* Create a organized assets structure: `public/images/nokia/` and `public/images/metrics/`.
* Move `output_final_loss_curve.png` and `output_unet_matrix.png` into `public/images/metrics/`.
* Add the Nokia Logo SVG to `public/images/nokia-logo.svg`.

### 4. Build Core Layout & Global Theme (`src/app/`)
* **`layout.tsx`:** Define the global HTML structure, fonts (Inter or Nokia Pure if available), and global state.
* **Navigation Component:** Create a responsive, sticky Navbar featuring:
    * The **Nokia Logo** + a sleek **"Hackathon 2026"** glowing pill badge.
    * Links to Overview, Models, and Benchmarks.
    * A blur effect (`backdrop-blur-md`) for a premium modern feel.
* **Footer:** Standard Nokia-inspired minimalist footer celebrating the innovation of the Hackathon.

### 5. Develop Optimized Pages (App Router)

#### Overview Page (`src/app/page.tsx`)
* **Hero Section:** A powerful Nokia-branded banner introducing the **Nokia Hackathon Challenge**.
* **The Problem:** Contextualize the CSI feedback overhead bottleneck in 5G Advanced and 6G Massive MIMO systems.
* **The Solution:** Summary of how AI-driven deep learning models radically optimize feedback compression and channel prediction.

#### Models Page (`src/app/models/page.tsx`)
* Display the architectures developed during the hackathon using clean Tailwind Cards with tech-badges:
    * **Conv3D + CBAM (Advanced Hybrid):** Spatial-temporal feature extraction.
    * **Diffusion Model (DDPM):** Generative approach for realistic CSI reconstruction.
    * **CNN + Transformer:** Attention-based long-range dependency mapping.
    * **Baselines (GRU, ConvLSTM):** Standard sequential models used for performance benchmarking.

#### Benchmarks Page (`src/app/benchmarks/page.tsx`)
* **Leaderboard Table:** A beautifully styled interactive table showing NMSE (Normalized Mean Square Error) scores.
    * *Highlight the winner:* Diffusion (~0.0031 NMSE) and Conv3D (~0.0033 NMSE) with special Nokia Blue victory highlights.
* **Visualizations Section:** Use Next.js `<Image />` component (with automatic WebP conversion and aspect ratio defense) to render:
    * `output_final_loss_curve.png` (Training vs Validation convergence).
    * `output_unet_matrix.png` (CSI prediction accuracy matrix vs Ground Truth).

---

## Vercel Deployment & Optimization
* **Server-Side Rendering (SSR) & Static Site Generation (SSG):** Keep pages static where possible for instant load times.
* **Edge Network Ready:** Ensure zero heavy client-side computations for the layout.
* **Deployment:** Connect the repository to Vercel for automated CI/CD previews on every push.

---

## Verification & QA Checklist
* [ ] **Build Success:** Run `npm run build` locally to ensure zero TypeScript or Next.js compilation errors.
* [ ] **Routing & Navigation:** Verify seamless instant page transitions via Next.js `<Link>`.
* [ ] **Brand & Responsive Design:** Check Tailwind breakpoints and ensure the Nokia Blue identity looks stunning on mobile, tablet, and desktop.
* [ ] **Image Optimization:** Confirm that the benchmark images use Next.js `next/image` and load efficiently without layout shifts (CLS).
* [ ] **Vercel Readiness:** Verify `vercel.json` or project settings match production requirements.