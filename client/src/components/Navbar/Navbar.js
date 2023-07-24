import React, { useState } from 'react';
import './Navbar.css';

const Navbar = ({ onDashboardClick, onHomeClick, onAboutClick, onContactClick }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [authenticated, setAuthenticated] = useState(false);

  const handleDashboardClick = (event) => {
    event.preventDefault();
    if (authenticated) {
      onDashboardClick();
    } else {
      alert('Authentication failed. Please provide valid credentials.');
    }
  };

  const handleHomeClick = (event) => {
    event.preventDefault();
    onHomeClick();
  };

  const handleContactClick = (event) => {
    event.preventDefault();
    onContactClick();
  };

  const handleUsernameChange = (event) => {
    setUsername(event.target.value);
  };

  const handlePasswordChange = (event) => {
    setPassword(event.target.value);
  };

  const handleLogin = (event) => {
    event.preventDefault();
    // Perform authentication logic here
    if (username === 'gouri' && password === 'gouri') {
      setAuthenticated(true);
      alert('Authentication successful. You can now access the dashboard.');
    } else {
      alert('Authentication failed. Please provide valid credentials.');
    }
    setUsername('');
    setPassword('');
  };

  return (
    <nav className="navbar">
      <ul>
        <img src="./user.png" style={{ width: '100px' }} alt="User" />
        <br />
        <br />
        <li>
          <a href="#" onClick={handleHomeClick}>
            Home
          </a>
        </li>
        <li>
          <a href="#" onClick={handleContactClick}>
            Contact
          </a>
        </li>
      </ul>
      {!authenticated && (
        <div className="login-box">
          <p style={{ color: '#28a9e2', fontSize: '14px' }}>To access the dashboard:</p>
          <form onSubmit={handleLogin}>
            <input
              type="text"
              placeholder="Username"
              value={username}
              onChange={handleUsernameChange}
              required
            />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={handlePasswordChange}
              required
            />
            <button type="submit">Login</button>
          </form>
        </div>
      )}
      {authenticated && (
        <ul>
          <li>
            <a href="#" onClick={handleDashboardClick}>
              Dashboard
            </a>
          </li>
        </ul>
      )}
    </nav>
  );
};

export default Navbar;
