/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"JetBrains Mono"', '"Fira Code"', 'ui-monospace', 'monospace'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'ui-monospace', 'monospace'],
      },
      colors: {
        cyber: {
          50: "#e6fff0",
          100: "#ccffe0",
          200: "#99ffc2",
          300: "#66ffa3",
          400: "#33ff85",
          500: "#00ff66",
          600: "#00cc52",
          700: "#00993d",
          800: "#006629",
          900: "#003314",
        },
        neon: {
          pink: "#ff006e",
          purple: "#8338ec",
          blue: "#3a86ff",
          cyan: "#00f5ff",
          green: "#00ff41",
        },
      },
      boxShadow: {
        "neon-sm": "0 0 5px rgba(0,255,65,0.5)",
        "neon-md": "0 0 10px rgba(0,255,65,0.6), 0 0 20px rgba(0,255,65,0.3)",
        "neon-lg": "0 0 15px rgba(0,255,65,0.7), 0 0 30px rgba(0,255,65,0.4), 0 0 45px rgba(0,255,65,0.2)",
        "neon-pink": "0 0 10px rgba(255,0,110,0.6), 0 0 20px rgba(255,0,110,0.3)",
        "neon-blue": "0 0 10px rgba(58,134,255,0.6), 0 0 20px rgba(58,134,255,0.3)",
      },
      animation: {
        'glow': 'glow 2s ease-in-out infinite alternate',
        'scan': 'scan 8s linear infinite',
      },
      keyframes: {
        glow: {
          '0%': { textShadow: '0 0 5px rgba(0,255,65,0.5), 0 0 10px rgba(0,255,65,0.3)' },
          '100%': { textShadow: '0 0 10px rgba(0,255,65,0.8), 0 0 20px rgba(0,255,65,0.5), 0 0 30px rgba(0,255,65,0.3)' },
        },
        scan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
      },
    },
  },
  plugins: [],
};

