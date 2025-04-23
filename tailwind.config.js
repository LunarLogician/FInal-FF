/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {},
  },
  safelist: [
    'bg-blue-500',
    'bg-green-500',
    'bg-yellow-500',
    'bg-purple-500',
    'bg-red-500',
    'bg-gray-400',
    'bg-indigo-500',
    'bg-emerald-500',
    'bg-orange-500',
    'bg-teal-500',
    'h-2',
  ],
  plugins: [],
};
