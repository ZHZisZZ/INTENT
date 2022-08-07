import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Avatar from '@material-ui/core/Avatar';
import Chip from '@material-ui/core/Chip';
import FaceIcon from '@material-ui/icons/Face';
import DoneIcon from '@material-ui/icons/Done';
import {paper} from '../styles/index'
import { Paper, Typography } from '@material-ui/core';
const useStyles = makeStyles((theme) => ({
    box: {
      'paddingLeft': theme.spacing(5),
      textAlign: 'left',
    },
    word:{
        textAlign: 'left',
        fontWeight: 800
    },
    chip:{
        marginLeft: theme.spacing(2)
    }

}));


export default function Preference({preference, setPreference}) {
  const classes = useStyles();

    const handleDeletePref = (k, preference, setPreference) =>{
        const newPreference = {...preference};
        delete newPreference[k];
        setPreference(newPreference)
    }
  return (
    <div className={classes.box}>
        {Object.keys(preference).length===0  || <span className={classes.word}>Preference Settings:</span>}
        {
            Object.keys(preference).map((k)=>{
                if(preference[k].desired){
                    return (<Chip className={classes.chip} label={preference[k].label} key={k} onClick={()=>handleDeletePref(k,preference,setPreference)} color="primary" />
                    )
                }
                else{
                    return (<Chip className={classes.chip} label={preference[k].label} key = {k} onClick={()=>handleDeletePref(k,preference,setPreference)} color="secondary" />)
                }
            })
        }
    </div>
  );
}
