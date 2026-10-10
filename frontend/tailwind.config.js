/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#050a14',
          900: '#070d1a',
          850: '#0a1220',
          800: '#0d1628',
          700: '#122033',
          600: '#1a2c45',
        },
        charcoal: {
          900: '#11161d',
          800: '#161c25',
          700: '#1d242f',
          600: '#26303d',
        },
        cyan: {
          brand: '#22d3ee',
          dim: '#0e7490',
        },
        azure: {
          brand: '#3b82f6',
          dim: '#1d4ed8',
        },
        risk: {
          healthy: '#10b981',
          watch: '#eab308',
          high: '#f97316',
          critical: '#ef4444',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      boxShadow: {
        card: '0 1px 2px rgba(0,0,0,0.4), 0 8px 24px -12px rgba(0,0,0,0.6)',
        glow: '0 0 0 1px rgba(34,211,238,0.18), 0 8px 32px -12px rgba(34,211,238,0.28)',
      },
      backgroundImage: {
        grid: 'linear-gradient(rgba(34,211,238,0.045) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,0.045) 1px, transparent 1px)',
      },
      keyframes: {
        pulseRing: {
          '0%': { boxShadow: '0 0 0 0 rgba(239,68,68,0.45)' },
          '70%': { boxShadow: '0 0 0 12px rgba(239,68,68,0)' },
          '100%': { boxShadow: '0 0 0 0 rgba(239,68,68,0)' },
        },
        sweep: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        pulseRing: 'pulseRing 2.2s cubic-bezier(0.4,0,0.6,1) infinite',
        sweep: 'sweep 2.4s linear infinite',
      },
    },
  },
  plugins: [],
}
