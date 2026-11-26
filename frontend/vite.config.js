import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Tauri loads the dev server from a fixed address (src-tauri/tauri.conf.json "devUrl").
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
  },
})
