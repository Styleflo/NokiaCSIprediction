# GEMINI.md - Nokia Hackathon Dashboard

## Project Overview
The **Nokia Hackathon Dashboard** is a high-performance Next.js application designed to showcase research results from a 5G Advanced & 6G hackathon. The core focus is on reducing **Channel State Information (CSI) feedback overhead** in Massive MIMO systems using Generative AI, specifically **Diffusion Models (DDPM)**.

### Tech Stack
- **Framework:** Next.js 16 (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS v4 (Custom "Sketch/Paper" aesthetic)
- **Icons:** Lucide React
- **Deployment:** Optimized for Vercel

### Architecture
- `src/app/`: Contains the main application pages and global layouts.
- `src/components/`: Reusable UI components (Navbar, Footer, Background effects).
- `public/images/`: Stores research metrics, logos, and other static assets.

## Building and Running
The project follows standard npm scripts for a Next.js application:

- **Development:** `npm run dev` (Runs on `http://localhost:3000`)
- **Build:** `npm run build` (Generates a production build)
- **Start:** `npm run start` (Starts the production server)
- **Lint:** `npm run lint` (Runs ESLint for code quality checks)

## Development Conventions

### Aesthetic & Styling
The project uses a unique **"Paper-like/Sketch"** visual style defined in `src/app/globals.css`. 
- **Colors:** Nokia Blue (`#1241C6`), Paper Off-White (`#F4F1EA`), and Graphite/Charcoal (`#2C2C2C`).
- **Patterns:** Radial paper grain background and sketchy card borders (`sketch-card`, `glass-card`).
- **Typography:** Prefers mono-spaced fonts for an architectural/technical feel.

### Component Guidelines
- **Server Components:** Use React Server Components by default for better performance.
- **Client Components:** Use `"use client"` only when interactive state or browser APIs are required (e.g., animations, background effects).
- **Icons:** Always use `lucide-react` for consistency.

### Data Representation
- Research metrics and benchmarks are currently hardcoded in `src/app/page.tsx` or displayed as optimized images using `next/image`.
- The **Leaderboard** highlights performance gains of Diffusion and Conv3D models over traditional baselines like ConvLSTM.

## Key Files
- `src/app/page.tsx`: The primary dashboard interface containing research results, interactive simulations, and methodology.
- `src/app/globals.css`: The source of truth for the project's custom theme and Tailwind v4 configuration.
- `src/components/CSISimulator.tsx`: Interactive network simulation comparing legacy FDD feedback vs AI-driven CSI prediction.
- `PLAN.md`: Contains the original implementation strategy and project goals.
