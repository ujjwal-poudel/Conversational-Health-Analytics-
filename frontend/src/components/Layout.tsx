import React from 'react'
import './Layout.css'

interface LayoutProps {
  children: React.ReactNode
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="layout">
      <nav className="navbar">
        <div className="navbar-container">
          <div className="navbar-brand">Mental Health App</div>
          <div className="navbar-links">
            <a href="#about" className="nav-link">About</a>
            <a href="#contact" className="nav-link">Contact</a>
          </div>
        </div>
      </nav>
      
      <main className="main-content">
        {children}
      </main>
      
      <footer className="footer">
        <p>Copyright © 2025</p>
      </footer>
    </div>
  )
}

export default Layout

