/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{html,js,svelte,ts}"],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "#10b981",
          light: "#34d399",
          dark: "#059669",
          glow: "rgba(16, 185, 129, 0.3)",
        },
        secondary: {
          DEFAULT: "#8b5cf6",
          light: "#a78bfa",
          dark: "#7c3aed",
        },
        accent: {
          blue: "#3b82f6",
          pink: "#ec4899",
          orange: "#f59e0b",
        },
        background: {
          primary: "#0a0e27",
          secondary: "#151b3b",
          tertiary: "#1e2749",
          card: "rgba(30, 39, 73, 0.6)",
          "card-hover": "rgba(30, 39, 73, 0.8)",
        },
        text: {
          primary: "#f9fafb",
          secondary: "#d1d5db",
          muted: "#9ca3af",
          dim: "#6b7280",
        },
        border: {
          DEFAULT: "rgba(139, 92, 246, 0.2)",
          hover: "rgba(139, 92, 246, 0.4)",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
      },
      animation: {
        "gradient-shift": "gradientShift 20s ease infinite",
        shimmer: "shimmer 2s infinite",
        "pulse-glow": "pulse-glow 2s ease-in-out infinite",
        float: "float 3s ease-in-out infinite",
        "fade-in": "fadeIn 0.5s ease-out",
      },
      keyframes: {
        gradientShift: {
          "0%, 100%": { transform: "translate(0, 0) rotate(0deg)" },
          "33%": { transform: "translate(5%, -5%) rotate(5deg)" },
          "66%": { transform: "translate(-5%, 5%) rotate(-5deg)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "pulse-glow": {
          "0%, 100%": {
            boxShadow:
              "0 0 10px var(--primary-glow), 0 0 20px var(--primary-glow)",
          },
          "50%": {
            boxShadow:
              "0 0 20px var(--primary-glow), 0 0 40px var(--primary-glow), 0 0 60px var(--primary-glow)",
          },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-10px)" },
        },
        fadeIn: {
          from: { opacity: "0", transform: "translateY(20px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      backdropBlur: {
        xs: "2px",
      },
      boxShadow: {
        "glow-sm": "0 0 10px rgba(16, 185, 129, 0.3)",
        "glow-md": "0 0 20px rgba(16, 185, 129, 0.4)",
        "glow-lg": "0 0 30px rgba(16, 185, 129, 0.5)",
        "purple-glow": "0 0 20px rgba(139, 92, 246, 0.4)",
        glass: "0 8px 32px 0 rgba(0, 0, 0, 0.37)",
      },
    },
  },
  plugins: [],
};
