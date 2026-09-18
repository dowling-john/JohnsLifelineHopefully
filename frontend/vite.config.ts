import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev-only: the FastAPI backend (decision_engine_main.py) serves /api and
// /ws on :8000. In production the built app is served BY that same FastAPI
// process (see ui_server.py's index()/StaticFiles mount), so this proxy
// only matters for `npm run dev`.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
})
