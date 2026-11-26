import React from 'react';

export default function Logo({ className = "w-10 h-10" }) {
  return (
    <svg 
      xmlns="http://www.w3.org/2000/svg" 
      viewBox="0 0 120 120" 
      className={`${className} shadow-lg shadow-blue-500/20 rounded-[22px]`}
      fill="none"
    >
      {/* Modern Blue Gradient Background */}
      <defs>
        <linearGradient id="documind-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#3b82f6" /> {/* Tailwind blue-500 */}
          <stop offset="100%" stopColor="#1d4ed8" /> {/* Tailwind blue-700 */}
        </linearGradient>
        <linearGradient id="accent-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#ffffff" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#ffffff" stopOpacity="0.4" />
        </linearGradient>
      </defs>

      {/* Main Squircle Background */}
      <rect width="120" height="120" rx="28" fill="url(#documind-grad)" />

      {/* Abstract 'D' / Document Shape (Glassmorphism effect) */}
      <path 
        d="M35 30H65C81.5685 30 95 43.4315 95 60C95 76.5685 81.5685 90 65 90H35V30Z" 
        fill="url(#accent-grad)"
      />
      
      {/* Inner cutout for the 'D' */}
      <path 
        d="M50 45H65C73.2843 45 80 51.7157 80 60C80 68.2843 73.2843 75 65 75H50V45Z" 
        fill="url(#documind-grad)"
      />

      {/* Semantic AI Node (The "Mind" part) */}
      <circle cx="65" cy="60" r="6" fill="white" />
      
      {/* Subtle connecting lines representing network/context */}
      <path d="M65 60L50 30" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeOpacity="0.6"/>
      <path d="M65 60L50 90" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeOpacity="0.6"/>
      
      {/* Sparkle / AI accent */}
      <path d="M90 25L92 30L97 32L92 34L90 39L88 34L83 32L88 30L90 25Z" fill="white" fillOpacity="0.8" />
    </svg>
  );
}