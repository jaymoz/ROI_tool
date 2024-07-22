import React, { useState, useEffect } from 'react';
import './Dashboard_sidebar.css';
import arrow_key from '../images/left_arrow.png';

function Dashboard_sidebar(){
    const [sideBarOpen, setSideBar] = useState(false);

    const handleSideBarButton = () =>{
        if (sideBarOpen){
            setSideBar(false);
        }
        else{
            setSideBar(true);
        }
    }

    useEffect(()=>{
        let sidebar_elem = document.querySelector('.Dashboard_sidebar');
        let sidebar_img = sidebar_elem.lastElementChild;
        let rect = sidebar_elem.getBoundingClientRect();

        if (sideBarOpen){
            sidebar_elem.style.transform= `translateX(0%)`;
            sidebar_img.style.transform = 'rotate(0deg)';
        }
        else{
            sidebar_elem.style.transform= `translateX(-${rect.right-40}px)`;
            sidebar_img.style.transform = 'rotate(180deg)';
        }
    },[sideBarOpen]);

    return(
        <div className='Dashboard_sidebar sidebar_visible'>
            <div className='sidebar_section sec1'>
                <h2>File Input</h2>
            </div>
            <div className='sidebar_section sec2'>
                <h2>ML Model</h2>
            </div>
            <div className='sidebar_section sec3'>
                <h2>ROI Analysis</h2>
            </div>
            <img src={arrow_key} alt="Cross" className='modal-cross-image' onClick={handleSideBarButton} />
        </div>
    );
}

export default Dashboard_sidebar;


