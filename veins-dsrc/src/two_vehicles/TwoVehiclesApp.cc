//
// Copyright (C) 2016 David Eckhoff <david.eckhoff@fau.de>
//
// Documentation for these modules is at http://veins.car2x.org/
//
// SPDX-License-Identifier: GPL-2.0-or-later
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//

#include "two_vehicles/TwoVehiclesApp.h"

#include "two_vehicles/TwoVehiclesMessage_m.h"

using namespace veins;

Define_Module(veins::TwoVehiclesApp);

void TwoVehiclesApp::initialize(int stage)
{
    DemoBaseApplLayer::initialize(stage);
    if (stage == 0) {
        // Initializing members and pointers of your application goes here
        EV << "Initializing " << par("appName").stringValue() << std::endl;
    }
    else if (stage == 1) {
        // Initializing members that require initialized other modules goes here
        EV << "Vehicle ID is " << myId << std::endl;
        gradientHash = myId;
    }
}

void TwoVehiclesApp::finish()
{
    DemoBaseApplLayer::finish();
    // statistics recording goes here
}

void TwoVehiclesApp::onBSM(DemoSafetyMessage* bsm)
{
    // Your application has received a beacon message from another car or RSU
    // code for handling the message goes here
    EV << "Vehicle " << myId << " received BSM with position " << bsm->getSenderPos() << std::endl;
    
    TwoVehiclesMessage* wsm = new TwoVehiclesMessage("gradients");  // Make message green
    populateWSM(wsm);
    wsm->setSenderAddress(myId);
    wsm->setGradientHash(myId);
    sendDown(wsm);
}

void TwoVehiclesApp::onWSM(BaseFrame1609_4* wsm)
{
    // Your application has received a data message from another car or RSU
    // code for handling the message goes here, see TraciDemo11p.cc for examples
    if (TwoVehiclesMessage* tvm = dynamic_cast<TwoVehiclesMessage*>(wsm)) {
        // Set vehicle to green
        findHost()->getDisplayString().setTagArg("i", 1, "green");
        EV << "Vehicle " << myId << " received WSM from senderAddress " << tvm->getSenderAddress()
           << " with gradientHash " << tvm->getGradientHash() << std::endl;
    }
}

void TwoVehiclesApp::onWSA(DemoServiceAdvertisment* wsa)
{
    // Your application has received a service advertisement from another car or RSU
    // code for handling the message goes here, see TraciDemo11p.cc for examples
}

void TwoVehiclesApp::handleSelfMsg(cMessage* msg)
{
    DemoBaseApplLayer::handleSelfMsg(msg);
    // this method is for self messages (mostly timers)
    // it is important to call the DemoBaseApplLayer function for BSM and WSM transmission
}

void TwoVehiclesApp::handlePositionUpdate(cObject* obj)
{
    DemoBaseApplLayer::handlePositionUpdate(obj);
    // the vehicle has moved. Code that reacts to new positions goes here.
    // member variables such as currentPosition and currentSpeed are updated in the parent class
}
