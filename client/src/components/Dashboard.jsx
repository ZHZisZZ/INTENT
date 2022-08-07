import {
    Grid,
    AppBar,
    Toolbar,
    IconButton,
    Typography,
    Button,
    Switch,
    FormGroup,
    FormControlLabel,
    Snackbar, InputAdornment, CircularProgress
} from '@material-ui/core';
import {createContext, useEffect, useRef, useState} from "react"
import ControlPanel from './ControlPanel.jsx'
import ValidationPanel from './ValidationPanel.jsx'

import TextField from '@material-ui/core/TextField';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogTitle from '@material-ui/core/DialogTitle';

import axios from "axios";
import * as React from "react";
import {Alert} from "@material-ui/lab";
export const BaselineContext = createContext();

axios.defaults.baseURL = process.env.REACT_APP_API_BASE_URL;

const Dashboard = (props) => {
    const [synthesizedSolutions, setSynthesizedSolutions] = useState({});
    const [userExpressions, setUserExpressions] = useState([]);
    const [solutions, setSolutions] = useState({});
    useEffect(() => {
        const newSolutions = {}
        for (let i = 0; i < Object.keys(synthesizedSolutions).length; i++) {
            newSolutions['solution' + i] = synthesizedSolutions['solution' + i]
        }
        for (let i = 0; i < userExpressions.length; i++) {
            newSolutions['solution' + (Object.keys(synthesizedSolutions).length + i)] = {expression: userExpressions[i]}
        }
        setSolutions(newSolutions)
    }, [synthesizedSolutions, userExpressions])
    const [preference, setPreference] = useState({});
    const [synthesizing, setSynthesizing] = useState(false);
    const [validating, setValidating] = useState(false);
    const [validations, setValidations] = useState({});
    const [synthesisSessionId, setSynthesisSessionId] = useState(null);
    // const [baseline, setBaseline] = useState(false);
    // const toggleBaseline = ()=>{
    //     setBaseline(!baseline);
    // };
    const [allInputTensors, setAllInputTensors] = useState([[{
        index: 0,
        shape: [2, 2],
        stringValue: '[[1,2],[3,4]]',
        value: [[1, 2], [3, 4]]
    }]]);
    const [allOutputTensors, setAllOutputTensors] = useState([{
        shape: [2, 2],
        stringValue: '[[1,3],[2,4]]',
        value: [[1, 3], [2, 4]]
    }]);
    const [selectedTestCase, setSelectedTestCase] = useState(0);
    const [selectedSolution, setSelectedSolution] = useState(0);

    const [addSolutionDialogOpen, setAddSolutionDialogOpen] = useState(false);
    const [editSolutionDialogOpen, setEditSolutionDialogOpen] = useState(false);
    const [editingSolutionIndex, setEditingSolutionIndex] = useState(-1);
    const [editingSolution, setEditingSolution] = useState('');
    const [editingSolutionError, setEditingSolutionError] = useState(null);
    const [validatingEditingSolution, setValidatingEditingSolution] = useState(false);
    const [errorOccurredSnackbarOpen, setErrorSnackbarOpen] = useState(false);
    const [errorOccurredSnackbarMessage, setErrorSnackbarMessage] = useState('');
    const [editingSolutionValidations, setEditingSolutionValidations] = useState({});

    const baseline = props.match.path === '/alpha';
    const convertTestCases = () => {
        const result = {};
        let convertToObjects = (pairs) => {
            let root = {};
            for (let i = 0; i < pairs.length; i++) {
                let tensors = pairs[i];
                let obj = {};
                for (let j = 0; j < tensors.length; j++) {
                    obj[`${j + 1}`] = tensors[j].stringValue;
                }
                root[`${i + 1}`] = obj;
            }
            return root;
        }
        result.inputs = convertToObjects(allInputTensors);
        result.outputs = convertToObjects(allOutputTensors.map(tensor => [tensor]));
        return result;
    }
    const addUserExpression = (expression) => {
        setUserExpressions([...userExpressions, expression])
        setSelectedSolution(Object.keys(synthesizedSolutions).length + userExpressions.length) // No off-by-one
    };
    const editUserExpression = (index, expression) => {
        index -= Object.keys(synthesizedSolutions).length
        setUserExpressions([...userExpressions.slice(0, index), expression, ...userExpressions.slice(index + 1)])
        setSelectedSolution(Object.keys(synthesizedSolutions).length + index)
    };
    const removeSolution = (index) => {
        if (index < Object.keys(synthesizedSolutions).length) {
            const newSynthesizedSolutions = {}
            for (let i = 0; i < Object.keys(synthesizedSolutions).length; i++) {
                if (i === index) continue;
                newSynthesizedSolutions['solution' + (i > index ? i - 1 : i)] = synthesizedSolutions['solution' + i]
            }
            console.log(newSynthesizedSolutions)
            setSynthesizedSolutions(newSynthesizedSolutions)
            // const newValidations = {}
            // for (let i = 0; i < Object.keys(validations).length; i++) {
            //     newValidations['example' + i] = {}
            //     for (let j = 0; j < Object.keys(solutions).length; j++) {
            //         if (j === selectedSolution) continue;
            //         newValidations['example' + i]['solution' + (j > selectedSolution ? j - 1 : j)] = validations['example' + i]['solution' + j]
            //     }
            // }
        } else {
            setUserExpressions([...userExpressions.slice(0, index - Object.keys(synthesizedSolutions).length), ...userExpressions.slice(index - Object.keys(synthesizedSolutions).length + 1)])
        }
        setValidations({})
    }
    const clearSolutions = () => {
        setSynthesizedSolutions({})
        setUserExpressions([])
        setValidations({})
    }
    const validateTestCases = () => {
        return allInputTensors.every(tensors => tensors.every(it => it.shape !== null)) && allOutputTensors.every(it => it.shape !== null);
    }
    const validateSolutions = () => {
        if (!validateTestCases()) {
            // TODO: Set error state
            return;
        }
        const postData = convertTestCases()
        postData.session_id = synthesisSessionId
        postData.solutions = {}
        console.log(solutions)
        for (let i = 0; i < Object.keys(solutions).length; i++) {
            postData.solutions['solution' + i] = solutions['solution' + i].expression
        }
        if (Object.keys(postData.solutions).length === 0) {
            return;
        }
        console.log('Validating:')
        console.log(postData)
        setValidating(true)

        axios.post(`/validate`, postData)
            .then(res => {
                console.log('Validated:')
                console.log(res.data)
                setValidations(res.data)
            })
            .catch(e => {
                setErrorSnackbarMessage('Failed to validate.');
                setErrorSnackbarOpen(true);
                console.log(e)
            })
            .finally(() => {
                setValidating(false)
            })
    }
    const validateEditingSolution = () => {
        if (editingSolution.trim() === '') {
            setValidatingEditingSolution(false);
            return;
        }
        setValidatingEditingSolution(true);
        setEditingSolutionError(null);

        const postData = convertTestCases()
        postData.session_id = synthesisSessionId
        postData.solutions = {
            'solution0': editingSolution
        }
        axios.post(`/validate`, postData)
            .then(res => {
                console.log('Validated:')
                console.log(res.data)
                setEditingSolutionValidations(res.data)

                for (let i = 0; i < Object.keys(res.data).length; i++) {
                    let data = res.data['example' + i]['solution0']
                    if (data.exception) {
                        let parsedMessage = data.eval_output
                        if (parsedMessage.includes('unexpected EOF') || parsedMessage.includes('invalid syntax')) {
                            parsedMessage = 'invalid syntax'
                        } else {
                            console.log(parsedMessage)
                            const res = parsedMessage.match(/name \'(.*)+\' is not defined/)
                            if (res) {
                                const name = res[1]
                                console.log(name)
                                if (name.startsWith('in')) {
                                    let num = name.substring(2)
                                    if (Number.isInteger(Number(num))) {
                                        parsedMessage = `input tensor ${num} does not exist`
                                    }
                                }
                            }
                        }
                        setEditingSolutionError(`Exception when evaluating test case ${i + 1}: ` + parsedMessage)
                    }
                }
            })
            .catch(e => {
                setValidatingEditingSolution(false);
                setEditingSolutionError(e.toString())

                setErrorSnackbarMessage('Failed to validate.');
                setErrorSnackbarOpen(true);
                console.log(e)
            })
            .finally(() => {
                setValidatingEditingSolution(false)
            })
    }

    useEffect(() => {
        validateSolutions()
    }, [allInputTensors, allOutputTensors, solutions])
    useEffect(() => {
        setValidatingEditingSolution(true);
        const timeoutId = setTimeout(() => validateEditingSolution(), 500);
        return () => clearTimeout(timeoutId);
    }, [editingSolution])

    return (
        <div className="App" style={{overflow: 'hidden'}}>
            <BaselineContext.Provider value = {baseline}>
                <AppBar position="relative">
                    <Toolbar>
                        <Typography style={{fontWeight: 'bold', flex: 1}} variant="h6" color="inherit">
                        INTENT
                        </Typography>
                        {/* <FormGroup>
                            <FormControlLabel control={<Switch defaultChecked onChange={toggleBaseline} />} label={(baseline)? 'Baseline' : 'Features'} />
                        </FormGroup> */}
                    </Toolbar>
                </AppBar>
                <Grid container>
                    <Grid item xs={4}>
                        <ControlPanel setValidations={setValidations} synthesisSessionId={synthesisSessionId} setSynthesisSessionId={setSynthesisSessionId} convertTestCases={convertTestCases} allInputTensors={allInputTensors} setAllInputTensors={setAllInputTensors} allOutputTensors={allOutputTensors} setAllOutputTensors={setAllOutputTensors} validateSolutions={validateSolutions} validations={validations} setSynthesizedSolutions={setSynthesizedSolutions} selectedTestCase={selectedTestCase} selectedSolution={selectedSolution} setSelectedTestCase={setSelectedTestCase} preference={preference} setPreference={setPreference} synthesizing={synthesizing} setSynthesizing={setSynthesizing} setErrorSnackbarOpen={setErrorSnackbarOpen} setErrorSnackbarMessage={setErrorSnackbarMessage}/>
                    </Grid>
                    <Grid item xs={8}>
                        <ValidationPanel
                                         allOutputTensors={allOutputTensors}
                                         removeSolution={removeSolution}
                                         clearSolutions={clearSolutions}
                                         setEditingSolution={setEditingSolution} setEditingSolutionIndex={setEditingSolutionIndex}
                                         setAddSolutionDialogOpen={v => {
                                             if (v && !validateTestCases()) {
                                                 setErrorSnackbarMessage('Errors are found in the test cases. Please fix them before continuing.')
                                                 setErrorSnackbarOpen(true);
                                             } else {
                                                 setEditingSolutionError(null)
                                                 setAddSolutionDialogOpen(v)
                                             }
                                         }}
                                         setEditSolutionDialogOpen={v => {
                                             if (v && !validateTestCases()) {
                                                 setErrorSnackbarMessage('Errors are found in the test cases. Please fix them before continuing.')
                                                 setErrorSnackbarOpen(true);
                                             } else {
                                                 setEditingSolutionError(null)
                                                 setEditSolutionDialogOpen(v)
                                             }
                                         }}
                                         solutions={solutions} setSolutions={setSolutions} validating={validating} validations={validations} setValidations={setValidations} selectedTestCase={selectedTestCase} selectedSolution={selectedSolution} setSelectedSolution={setSelectedSolution} preference={preference} setPreference={setPreference} synthesizing={synthesizing} setSynthesizing={setSynthesizing} />
                    </Grid>
                </Grid>
                <Dialog maxWidth={'sm'} fullWidth={true} open={addSolutionDialogOpen || editSolutionDialogOpen} aria-labelledby="form-dialog-title">
                    <DialogTitle id="form-dialog-title">{addSolutionDialogOpen ? "Add a new tensor transformation expression" : "Edit tensor transformation expression"}</DialogTitle>
                    <DialogContent>
                        <DialogContentText>
                            You may refer to input tensors as <code>in1</code>, <code>in2</code>, etc.
                        </DialogContentText>
                        <div className={"code-edit"}>
                            <TextField
                                fullWidth
                                autoFocus
                                variant="outlined"
                                label="Expression"
                                value={editingSolution}
                                onChange={(e) => setEditingSolution(e.target.value)}
                                InputProps={{
                                    endAdornment: <InputAdornment position="end">{validatingEditingSolution && <CircularProgress color="primary" size="1.25rem" />}</InputAdornment>
                                }}
                                error={editingSolutionError !== null}
                                helperText={editingSolutionError}
                            />
                        </div>
                    </DialogContent>
                    <DialogActions>
                        <Button onClick={() => {
                            if (addSolutionDialogOpen) {
                                setAddSolutionDialogOpen(false)
                            } else {
                                setEditSolutionDialogOpen(false)
                            }
                        }} color="primary">
                            Cancel
                        </Button>
                        <Button disabled={editingSolution.trim() === ''} onClick={() => {
                            if (addSolutionDialogOpen) {
                                setAddSolutionDialogOpen(false)
                                addUserExpression(editingSolution)
                            } else {
                                setEditSolutionDialogOpen(false)
                                editUserExpression(editingSolutionIndex, editingSolution)
                            }
                        }} color="primary">
                            OK
                        </Button>
                    </DialogActions>
                </Dialog>
                <Snackbar autoHideDuration={6000} anchorOrigin={{ vertical: 'top', horizontal: 'center' }} open={errorOccurredSnackbarOpen} onClose={() => setErrorSnackbarOpen(false)}>
                    <Alert onClose={() => setErrorSnackbarOpen(false)} severity="error">{errorOccurredSnackbarMessage}</Alert>
                </Snackbar>
            </BaselineContext.Provider>
        </div>
    );
}

export default Dashboard;
