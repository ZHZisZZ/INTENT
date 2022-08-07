import {
    Grid,
    Button,
    Paper,
    makeStyles,
    Container,
    Box,
    Typography,
    CircularProgress,
    Tooltip, IconButton,
    Switch
} from '@material-ui/core';
import Solutions from './Solutions';
import Graph from './Graph';
import {useContext, useState} from 'react';
import CardContent from "@material-ui/core/CardContent";
import Card from "@material-ui/core/Card";
import * as React from "react";
import { BaselineContext } from './Dashboard';
import AddIcon from '@material-ui/icons/Add';
import RemoveIcon from '@material-ui/icons/Remove';
import ClearAllIcon from '@material-ui/icons/ClearAll';
const useStyles = makeStyles((theme) => ({
    solutions:{
        height: '294px',
    },
    diagram:{
        height: 'calc(100vh - 64px - 294px - 3rem)',
        overflow: 'auto',
    }
}));
const getExpHeight = (showExpr) => { return  (showExpr) ? '294px' : '50px' }
const ValidationPanel = ({allOutputTensors, solutions, removeSolution, clearSolutions, validating, validations, setValidations, selectedTestCase, selectedSolution, setSelectedSolution, preference, setPreference, synthesizing, setSynthesizing, setEditingSolution, setEditingSolutionIndex, setAddSolutionDialogOpen, setEditSolutionDialogOpen}) => {
    const classes = useStyles();
    const baseline = useContext(BaselineContext)
    const [showExpr, setShowExpr] = useState(true);
    return (
        <Box style={{overflow: 'hidden', height: 'calc(100vh - 64px)'}}>
            <Card style={{margin: '1rem', height: getExpHeight(showExpr)}} variant="outlined">
                <CardContent style={{padding: '0rem'}}>
                    <div style={{display: 'flex'}}>
                        <Typography style={{
                            fontSize: 14,
                            fontWeight: 'bold',
                            textTransform: "uppercase",
                            marginBottom: '0rem',
                            display: 'flex',
                            alignItems: 'center',
                            columnGap: '0.75rem',
                            padding: '1rem',
                            flex: 1
                        }} color="primary" gutterBottom>
                            Tensor Transformation Expressions {validating && <CircularProgress color="inherit" size="1.25rem" />}
                        </Typography>
                        {!baseline && <Box>
                            <Tooltip title="Show/Hide Expressions" aria-label="Show/Hide Expressions">
                            <Switch
                                checked={showExpr}
                                onChange={() => setShowExpr(!showExpr)}
                                name="Show/Hide Expressions"
                                color="primary"
                            />
                                </Tooltip>
                            <Tooltip title="Add a tensor transformation expression" aria-label="Add a tensor transformation expression">
                                <IconButton aria-label="Add a tensor transformation expression" color="primary"
                                            disabled={!showExpr}
                                            onClick={() => {
                                                setEditingSolution('')
                                                setAddSolutionDialogOpen(true)
                                            }}>
                                    <AddIcon/>
                                </IconButton>
                            </Tooltip>
                            <Tooltip title="Remove selected tensor transformation expression" aria-label="Remove selected tensor transformation expression">
                                <IconButton disabled={selectedSolution >= Object.keys(solutions).length || !showExpr} aria-label="Remove selected tensor transformation expression"
                                            color="primary" onClick={() => removeSolution(selectedSolution)}>
                                    <RemoveIcon/>
                                </IconButton>
                            </Tooltip>
                            <Tooltip title="Clear all tensor transformation expressions" aria-label="Clear all tensor transformation expressions">
                                <IconButton disabled={Object.keys(solutions).length === 0 || !showExpr} aria-label="Clear all tensor transformation expressions"
                                            color="primary" onClick={() => clearSolutions()}>
                                    <ClearAllIcon/>
                                </IconButton>
                            </Tooltip>
                        </Box>}
                    </div>
                    {showExpr && <Solutions setEditingSolution={setEditingSolution} setEditingSolutionIndex={setEditingSolutionIndex} setAddSolutionDialogOpen={setAddSolutionDialogOpen} setEditSolutionDialogOpen={setEditSolutionDialogOpen} solutions={solutions} validating={validating} validations={validations} setSelectedSolution={setSelectedSolution} selectedSolution={selectedSolution}/>}
                </CardContent>
            </Card>
            {!baseline && <Card style={{height: `calc(100vh - 64px - ${getExpHeight(showExpr)} - 3rem)`, margin: '1rem', overflow: 'auto'}} variant="outlined">
                <CardContent style={{padding: '1rem'}}>
                    <Typography style={{
                        fontSize: 14,
                        fontWeight: 'bold',
                        textTransform: "uppercase",
                        marginBottom: '1.5rem'
                    }} color="primary" gutterBottom>
                        Dataflow-based Explanation
                    </Typography>

                <Graph allOutputTensors={allOutputTensors} validations={validations} selectedTestCase={selectedTestCase} selectedSolution={selectedSolution} preference={preference} setPreference={setPreference} />

                </CardContent>
            </Card>
            }

        </Box>
    );
}
 
export default ValidationPanel;
