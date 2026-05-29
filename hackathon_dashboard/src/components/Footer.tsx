export default function Footer() {
  return (
    <footer className="bg-background border-t border-accent-muted">
      <div className="max-w-5xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col items-center gap-4 text-center">
          <div className="text-[10px] font-mono tracking-widest text-text-muted uppercase">
            © 2026 Nokia Research • Hackathon Edition
          </div>
          <div className="text-xs text-text-muted italic">
            &quot;Innovation in Channel State Information Prediction for Next-Generation Wireless Networks&quot;
          </div>
          <div className="mt-4 h-px w-8 bg-accent-muted" />
        </div>
      </div>
    </footer>
  );
}
