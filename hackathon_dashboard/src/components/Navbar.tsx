import Link from 'next/link';
import Image from 'next/image';

export default function Navbar() {
  return (
    <nav className="sticky top-0 z-50 w-full bg-background/80 backdrop-blur-md border-b border-accent-muted">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-14 items-center">
          <div className="flex items-center gap-6">
            <Link href="/" className="flex items-center text-foreground hover:text-nokia-blue transition-colors">
              <Image 
                src="/images/nokia-refreshed.svg" 
                alt="Nokia Logo" 
                width={180} 
                height={42} 
                className="h-9 w-auto"
              />
            </Link>
            <div className="h-4 w-px bg-accent-muted hidden sm:block" />
            <span className="text-[10px] font-mono tracking-widest text-text-muted uppercase hidden sm:block">
              Research Publication • Hackathon 2026
            </span>
          </div>
          <div className="flex items-center space-x-6">
            <Link href="#abstract" className="text-xs font-medium text-text-muted hover:text-nokia-blue transition-colors">
              Abstract
            </Link>
            <Link href="#results" className="text-xs font-medium text-text-muted hover:text-nokia-blue transition-colors">
              Results
            </Link>
            <Link href="#methodology" className="text-xs font-medium text-text-muted hover:text-nokia-blue transition-colors">
              Methodology
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}
