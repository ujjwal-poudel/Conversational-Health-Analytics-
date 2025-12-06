import { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import Home from './pages/Home';
import ModeSelect from './pages/ModeSelect';
import TextChat from './pages/TextChat';
import VoiceChat from './pages/VoiceChat';
import Results from './pages/Results';
import './App.css';

// Wrapper component to handle refresh-to-home
function NavigationGuard({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    // Check if this is a fresh page load (refresh)
    const isInAppNavigation = sessionStorage.getItem('inAppNavigation');

    if (!isInAppNavigation && location.pathname !== '/') {
      // Fresh load/refresh on a non-home page -> redirect to home
      navigate('/', { replace: true });
    }

    // Mark that we're now in the app
    sessionStorage.setItem('inAppNavigation', 'true');
  }, []);

  return <>{children}</>;
}

function App() {
  return (
    <Router>
      <NavigationGuard>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/mode" element={<ModeSelect />} />
          <Route path="/chat" element={<TextChat />} />
          <Route path="/audio" element={<VoiceChat />} />
          <Route path="/results" element={<Results />} />
        </Routes>
      </NavigationGuard>
    </Router>
  );
}

export default App;

