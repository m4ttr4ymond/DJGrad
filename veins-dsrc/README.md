## Dependencies
1. `veins 5.1`
2. `OMNeT++ 5.7`
3. `SUMO v1.10.0`

## Launching OMNeT++
1. In your Terminal window, navigate to folder where OMNeT++ was installed (e.g., `~/omnetpp-5.7/`)
2. Change to `bin/` sub-directory
3. Run `./omnetpp`

## Adding veins-dsrc to OMNeT++ Workspace
1. `File` > `Import...`
2. Select `Existing Projects into Workspace` under `General/` > Click `Next`
3. Click `Directory...`
4. Navigate to `veins-dsrc/` folder > Click `Open`
   - `veins-dsrc` should be automatically selected as an Eclipse project now
5. Click `Finish`

## Adding veins to OMNeT++ Workspace
1. `File` > `Import...`
2. Select `Existing Projects into Workspace` under `General/` > Click `Next`
3. Click `Directory...`
4. Navigate to folder where you downloaded veins > Click `Open`
5. Click `Finish`
6. Right-click on `veins` in the workspace > `Build Project`

## Linking veins-dsrc with veins
1. Right-click on `veins-dsrc` in the workspace > `Properties`
2. Select `Project References`
3. Check the box next to `veins`
4. Select `Makemake` under `OMNeT++`
5. Select the `src` file
6. Click `Options...`
7. Go to the `Compile` tab
8. Check the box next to `Add include paths exported from referenced projects` if it is not selected already > `OK`
9. Click `Apply and Close`

## Build veins-dsrc
1. Right-click on `veins-dsrc` in workspace > `Build Project`

## Run veins-dsrc
### Connect SUMO to OMNeT++
1. In your Terminal window, navigate to the folder where you installed `veins`
2. Change to the `bin/` sub-folder
3. Run `./veins_launchd -vv -c sumo`
   - You should see the message `Listening on port 9999`

### Run a veins-dsrc Simulation
1. Expand the `simultations/` folder in the workspace
2. Expand one of the sub-folders
3. Right-click its `omnetpp.ini` file > `Run As` > `OMNeT++ Simulation`
4. Select any option under `Config name` > `OK`
5. Click the `RUN` button
