import React, { useEffect, useState } from 'react';
import './Navbar.css';
import Modal from 'react-modal';
import { Link,useNavigate  } from 'react-router-dom';
import crossImage from '../images/cross.png';

Modal.setAppElement('#root');

const Navbar = () => {
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [authenticated, setAuthenticated] = useState(false);
  const [modalIsOpen, setModalIsOpen] = useState(false);

  useEffect(
    ()=>{
      let login_value = localStorage.getItem('isLoggedIn');
      if (login_value){
        console.log('user is logged in');
      }
      if (login_value === 'true') {
        setAuthenticated(true);
        navigate('/dashboard');
      }
    },[]
  );

  // Open the modal
  const openModal = () => setModalIsOpen(true);

  // Close the modal
  const closeModal = () => setModalIsOpen(false);

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
      setModalIsOpen(false);
      navigate('/dashboard');
    }
    setUsername('');
    setPassword('');
    localStorage.setItem('isLoggedIn', 'true');
  };

  const handleLogout = (event)=>{
    event.preventDefault();
    setAuthenticated(false);
    localStorage.removeItem('isLoggedIn');
    navigate('/');
  };

  useEffect(()=>{
    let root_elem = document.querySelector('#root');
    if (modalIsOpen){
      root_elem.classList.add('blurred-content');
    }else{
      root_elem.classList.remove('blurred-content');
    }
  },[modalIsOpen]);

  return (
    <div className='navbar'>
      {!authenticated ? (<nav className="navbar">
        <div className='navbar-button'> 
          <Link to={"/"}>HomePage</Link>
          <Link to={"/contact"}>Contacts</Link>
        </div>
        <div className='login-button' onClick={openModal}>Login</div>
        <Modal
          isOpen={modalIsOpen}
          onRequestClose={closeModal}
          contentLabel="Example Modal"
          className="modal"
          overlayClassName="overlay"
        >
          <form className='modal-form' onSubmit={handleLogin}>
            <div className="form-group">
              <label htmlFor="username">Username</label>
              <input
                type="text"
                id="username"
                name="username"
                value={username}
                onChange={handleUsernameChange}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="password">Password</label>
              <input
                type="password"
                id="password"
                name="password"
                value={password}
                onChange={handlePasswordChange}
                required
              />
            </div>

            <button type="submit">Login</button>
          </form>
          <img src={crossImage} alt="Cross" className='modal-cross-image' onClick={closeModal} />
        </Modal>
      </nav>) : 
      (<nav className="side-bar navbar">
        <div className='navbar-button'> 
          <Link to={"/dashboard"}>File Upload</Link>
          <Link to={"/dashboard/ml-analysis"}>ML Analytics</Link>
          <Link to={"/dashboard/roi-analysis"}>ROI Analytics</Link>
        </div>
        <div className='login-button' onClick={handleLogout}>Logout</div>
      </nav>)}
    </div>
  );
};

export default Navbar;
