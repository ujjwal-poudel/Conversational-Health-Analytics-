import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3001,  // Changed from 3000 to 3001
    proxy: {
      '/api': {
        target: 'http://localhost:8001',  // Changed from 8000 to 8001
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})

