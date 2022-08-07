import {makeStyles} from '@material-ui/core/styles';
import {
    Box,
    CircularProgress,
    IconButton,
    Paper,
    Snackbar,
    Table, TableBody, TableCell,
    TableContainer,
    TableHead, TableRow, Tooltip, Typography
} from "@material-ui/core";
import FileCopyIcon from '@material-ui/icons/FileCopy';
import {useState, useContext} from "react";
import Highlight from 'react-highlight'
import {Alert} from "@material-ui/lab";
import 'highlight.js/styles/atom-one-light.css'
import * as React from "react";
import CheckCircleIcon from '@material-ui/icons/CheckCircle';
import CancelIcon from '@material-ui/icons/Cancel';
import ComputerIcon from '@material-ui/icons/Computer';
import CreateIcon from '@material-ui/icons/Create';
import { BaselineContext } from './Dashboard';
const useStyles = makeStyles((theme) => ({
    root: {
        width: '100%',
        overflow: 'auto',
        height: '240px',
        backgroundColor: theme.palette.background.paper,
    },
}));
const Solutions = ({solutions, validating, validations, setSelectedSolution, selectedSolution, setEditingSolution, setEditingSolutionIndex, setAddSolutionDialogOpen, setEditSolutionDialogOpen}) => {
    const classes = useStyles();
    const baseline = useContext(BaselineContext)
    const [copiedToClipboardSnackbarOpen, setCopiedToClipboardSnackbarOpen] = useState(false);
    const formatExpressionScore = (index) => {
        if (!validations['example0'] || !validations['example0']['solution' + index]) {
            return <CircularProgress color="inherit" size="1.25rem" />
        }
        const passed = Object.keys(validations).filter(exampleKey => validations[exampleKey]['solution' + index].match).length
        const total = Object.keys(validations).length
        const failedCases = Object.keys(validations).filter(exampleKey => !validations[exampleKey]['solution' + index].match).map(it => Number(it.substring(7)) + 1)
        return <Tooltip title={passed === total ? "Passed all test cases" : "Failed on test case" + (failedCases.length === 1 ? "" : "s") + " " + failedCases.join(', ')}>
            <div style={{display: 'flex', columnGap: '0.25rem', justifyContent: 'center', alignItems: 'center'}}>
                {passed === total && <CheckCircleIcon htmlColor='#4caf50' />}
                {passed < total && <CancelIcon color={"error"} />}
                <Typography style={{
                    fontSize: 14,
                    fontWeight: 'bold',
                    textTransform: "uppercase",
                    color: passed < total ? '#f44336' : '#4caf50'
                }}>
                    {passed}/{total}
                </Typography>
            </div>
        </Tooltip>
    }
    return (
        <div className={classes.root}>
            <TableContainer elevation={0} square component={Paper}>
                <Table stickyHeader className={classes.table} size="small" aria-label="a dense table">
                    <TableHead>
                        <TableRow>
                            <TableCell>Expression</TableCell>
                            <TableCell align="center">Test results</TableCell>
                            <TableCell align="center">Actions</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {Object.keys(solutions).map((id, index) => [solutions[id], index]).map(([sol, index]) => (
                            <TableRow key={index}
                                      hover
                                      selected={selectedSolution === index}
                                      onClick={(event) => setSelectedSolution(index)}>
                                <TableCell component="th" scope="row" style={{fontFamily: 'monospace', fontSize: '14px'}}>
                                    <div style={{display: 'flex', columnGap: '0.5rem', justifyContent: 'flex-start', alignItems: 'center'}}>
                                        
                                        { sol.time &&
                                            <Tooltip title={`Synthesized by INTENT in ${sol.time} s`} aria-label="Synthesized by INTENT">
                                                <ComputerIcon color={"primary"}/>
                                            </Tooltip>
                                        }
                                        <code>{sol.expression}</code>
                                    </div>
                                </TableCell>
                                <TableCell align="center">
                                    {formatExpressionScore(index)}
                                </TableCell>
                                <TableCell align="center">
                                     <Box style={{display: 'flex', justifyContent: 'center', marginLeft: '-12px', columnGap: '12px'}}>
                                     {!baseline && <Tooltip title="Edit expression" aria-label="Edit expression">
                                            <IconButton edge="end" onClick={() => {
                                                // Synthesized solution?
                                                if (sol.time) {
                                                    // Make a copy
                                                    setEditingSolution(sol.expression)
                                                    setAddSolutionDialogOpen(true)
                                                } else {
                                                    setEditingSolutionIndex(index)
                                                    setEditingSolution(sol.expression)
                                                    setEditSolutionDialogOpen(true)
                                                }
                                            }}>
                                                <CreateIcon />
                                            </IconButton>
                                        </Tooltip>
                                    }
                                        <Tooltip title="Copy expression to clipboard" aria-label="copy expression to clipboard">
                                            <IconButton edge="end" onClick={() =>
                                                navigator.clipboard.writeText(sol.expression).then(() => {
                                                    setCopiedToClipboardSnackbarOpen(true)
                                                }, () => alert('Copy to clipboard failed.'))}>
                                                <FileCopyIcon />
                                            </IconButton>
                                        </Tooltip>
                                    </Box>
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
            <Snackbar autoHideDuration={6000} anchorOrigin={{ vertical: 'top', horizontal: 'center' }} open={copiedToClipboardSnackbarOpen} onClose={() => setCopiedToClipboardSnackbarOpen(false)}>
                <Alert onClose={() => setCopiedToClipboardSnackbarOpen(false)} severity="success">Code copied to clipboard!</Alert>
            </Snackbar>
        </div>
    );
}

export default Solutions;
