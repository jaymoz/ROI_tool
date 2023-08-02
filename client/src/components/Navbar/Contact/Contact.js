import React from 'react';
import './Contact.css';

const Contact = ({}) => {

  const professorName = "Dr. Gouri Deshpande";
  const uni = "University of Calgary";
  const dept = "Deptartment of Electrical and Software Engineering";
  const contactDetails = "Email: gouri.deshpande@ucalgary.ca";
  const professorImageURL = "./gouri.jpg"; 

  const ulStyle = {
    fontSize: '18px',
    color: 'gray',
    paddingLeft: '20px'
  };

  const students = [
    {
      name: "Noopur Zambare",
      institute: "",
      project: 'Mitacs Summer Intern 2023',
      imageUrl: "./noopur.jpg", 
    },
    {
      name: "Ammar Elzeftawy",
      institute: "",
      project: 'Schulich Summer Intern 2023',
      imageUrl: "./ammar.jpg", 
    },
    {
      name: "Zeeshan Chougle",
      institute: "",
      project: 'PURE Summer Intern 2023',
      imageUrl: "./zeeshan.jpg", 
    },
    {
      name: "Rishabh Ruhela",
      institute: "",
      project: 'Schulich Summer Intern 2023',
      imageUrl: "./rishabh.jpg", 
    },

  ];

  return (
    <div className="contact-container">
      <div className="professor-info">
        <img src={professorImageURL} alt="Professor" className="professor-image" />
        <div className="professor-details">
          <h2>{professorName}</h2>
          <strong><p style={{fontSize:'18px'}}>{uni}</p></strong>
    
          <ul style={ulStyle}>
            <li>{dept}</li>
            <li>{contactDetails}</li>
          </ul>

        </div>
      </div>

      <div className="students-list">
        
        <div className="students-tiles">
          <div className='conatct-text' style={{fontSize:'28px',fontWeight:'bold', color:'black', textAlign:'center',marginBottom:'20px'}}>Developed by :</div>
          {students.map((student, index) => (
            <div className="student-tile" key={index}>
              <div className="student-image">
                <img src={student.imageUrl} alt={`Student ${index + 1}`} className="student-image"/>
              </div>
              <div className="student-info">
              <strong style={{color:'black', fontSize:'20px'}}>{student.name}</strong>
                <ul style={{fontSize: '18px',color: 'gray',}}>       
                  <li>{student.project}</li>        
                  <li>{student.institute}</li>
                  
                </ul>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Contact;
